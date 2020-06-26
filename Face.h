#pragma once

#include <dlib/matrix.h>

#include <QVector>
#include <QRect>

class QImage;

struct Face
{
    /// For convenience, in case applications want to download it on demand
    /// Though dlib.net doesn't support https, so it's up to you
    static constexpr const char *shapePredictorFileName = "shape_predictor_5_face_landmarks.dat";
    static constexpr const char *shapePredictorUrl = "http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2";

    static constexpr const char *netFileName = "dlib_face_recognition_resnet_model_v1.dat";
    static constexpr const char *netUrl = "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2";

    // Identity descriptor of the face
    using Descriptor = dlib::matrix<float,0,1>;
    Descriptor descriptor;

    /// Rectangle in the image where it is
    QRect rectangle;

    QString imageId;

    QString name;

    static bool loadData();

    /// Returns a list of faces, the id should be a unique ID for the source
    /// image (e. g. file path)
    static QVector<Face> findFaces(const QImage &image, const QString &id);
};

