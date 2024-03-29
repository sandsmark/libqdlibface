#include "Face.h"

#include "Util.h"

#include <QStandardPaths>
#include <QImage>
#include <QDebug>

#include <dlib/image_processing/frontal_face_detector.h>

#include <dlib/dnn/layers.h>
#include <dlib/dnn/loss.h>
#include <dlib/dnn/input.h>

namespace
{
using namespace dlib;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = dlib::add_prev1<block<N,BN,1,dlib::tag1<SUBNET>>>;

template <template <int,template<typename>class,int,typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = dlib::add_prev2<dlib::avg_pool<2,2,2,2,dlib::skip1<dlib::tag2<block<N,BN,2,dlib::tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block  = BN<dlib::con<N,3,3,1,1,dlib::relu<BN<dlib::con<N,3,3,stride,stride,SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = dlib::relu<residual<block,N,dlib::affine,SUBNET>>;
template <int N, typename SUBNET> using ares_down = dlib::relu<residual_down<block,N,dlib::affine,SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256,SUBNET>;
template <typename SUBNET> using alevel1 = ares<256,ares<256,ares_down<256,SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128,ares<128,ares_down<128,SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64,ares<64,ares<64,ares_down<64,SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32,ares<32,ares<32,SUBNET>>>;

using anet_type = dlib::loss_metric<dlib::fc_no_bias<128,dlib::avg_pool_everything<
                            alevel0<
                            alevel1<
                            alevel2<
                            alevel3<
                            alevel4<
                            dlib::max_pool<3,3,2,2,dlib::relu<dlib::affine<dlib::con<32,7,7,2,2,
                            dlib::input_rgb_image_sized<150>
                            >>>>>>>>>>>>;


template <int N, typename SUBNET> using res       = relu<residual<block,N,bn_con,SUBNET>>;
template <int N, typename SUBNET> using res_down  = relu<residual_down<block,N,bn_con,SUBNET>>;
template <typename SUBNET> using level0 = res_down<256,SUBNET>;
template <typename SUBNET> using level1 = res<256,res<256,res_down<256,SUBNET>>>;
template <typename SUBNET> using level2 = res<128,res<128,res_down<128,SUBNET>>>;
template <typename SUBNET> using level3 = res<64,res<64,res<64,res_down<64,SUBNET>>>>;
template <typename SUBNET> using level4 = res<32,res<32,res<32,SUBNET>>>;

using net_type = loss_metric<fc_no_bias<128,avg_pool_everything<
                            level0<
                            level1<
                            level2<
                            level3<
                            level4<
                            max_pool<3,3,2,2,relu<bn_con<con<32,7,7,2,2,
                            input_rgb_image
                            >>>>>>>>>>>>;

struct Data {

    Data() :
        detector(dlib::get_frontal_face_detector())
    {
        const QString netPath = QStandardPaths::locate(
                QStandardPaths::AppDataLocation,
                Face::netFileName
            );

        if (netPath.isEmpty()) {
            qWarning() << "Failed to locate" << Face::netFileName;

            qDebug() << "It can be downloaded from"
                << Face::netUrl
                << "and decompressed into "
                << QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);

            qDebug() << "Or just complain to whomever supplied your application";
            return;
        }

        const QString shapePredictorPath = QStandardPaths::locate(
                QStandardPaths::AppDataLocation,
                Face::shapePredictorFileName
            );

        if (shapePredictorPath.isEmpty()) {
            qWarning() << "Failed to locate" << Face::shapePredictorFileName;
            qDebug() << "It can be downloaded from"
                << Face::shapePredictorUrl
                << "and decompressed into "
                << QStandardPaths::writableLocation(QStandardPaths::AppDataLocation);
            qDebug() << "Or just complain to whomever supplied your application";
            return;
        }

        dlib::deserialize(shapePredictorPath.toStdString()) >> shapePredictor;
        dlib::deserialize(netPath.toStdString()) >> metricNet;

        isValid = true;
    }

    bool isValid = false;

    dlib::frontal_face_detector detector;
    dlib::shape_predictor shapePredictor;
    net_type metricNet;

    static Data &instance() {
        static thread_local Data me;
        return me;
    }
};
}

bool Face::loadData()
{
    Data &data = Data::instance();
    return data.isValid;
}

QVector<Face> Face::findFaces(const QImage &qimage, const QString &id)
{
    Data &data = Data::instance();
    if (!data.isValid) {
        qWarning() << "Data failed to load";
        return {};
    }

    // QImage is COW, so not horribly inefficient
    const QImage sourceImage = (qimage.format() == QImage::Format_RGB888) ?
        qimage : qimage.convertToFormat(QImage::Format_RGB888);

    dlib::matrix<dlib::rgb_pixel> image(sourceImage.height(), sourceImage.width());

    const int bplDlib = dlib::width_step(image);
    if (bplDlib > sourceImage.bytesPerLine()) {
        qWarning() << "Something is wrong, bytes per line in qimage is too low";
        return {};
    }

    uchar *dlibData = reinterpret_cast<uchar*>(dlib::image_data(image));
    if (sourceImage.bytesPerLine() == bplDlib) {
        memcpy(dlibData, sourceImage.constBits(), dlib::image_size(image) * sizeof(dlib::rgb_pixel));
    } else {
        memset(dlibData, 0, dlib::image_size(image) * sizeof(dlib::rgb_pixel));

        for (int y=0; y<sourceImage.height(); y++) {
            memcpy(dlibData + bplDlib * y, sourceImage.scanLine(y), bplDlib);
        }
    }

    std::vector<dlib::matrix<dlib::rgb_pixel>> faces;
    QVector<QRect> rects;

    static const int chipSize = 150;
    static const float padding = 0.25;
    for (const dlib::rectangle &faceRect : data.detector(image)) {
        rects.append(dlibRectToQRect(faceRect));

        const dlib::full_object_detection shape = data.shapePredictor(image, faceRect);
        dlib::matrix<dlib::rgb_pixel> face;
        dlib::extract_image_chip(image, dlib::get_face_chip_details(shape, chipSize, padding), face);
        faces.push_back(std::move(face));
    }


    std::vector<dlib::matrix<float,0,1>> descriptors = data.metricNet(faces);
    assert(descriptors.size() == size_t(rects.size()));

    QVector<Face> ret(descriptors.size());
    for (size_t i=0; i<descriptors.size(); i++) {
        ret[i].descriptor = descriptors[i];
        ret[i].rectangle = rects[i];
        ret[i].imageId = id;
    }

    return ret;
}

