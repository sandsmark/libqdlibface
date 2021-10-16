#include "Database.h"
#include "Face.h"

#include <QDebug>

#include <dlib/clustering/chinese_whispers.h>

QString Database::findFast(const Face &face, double *score)
{
    if (m_centroidsOutdated) {
        updateCache();
    }

    if (m_centroids.empty()) {
        qWarning() << "No centroids loaded!";
        return {};
    }

    assert(m_centroidNames.size() == m_centroids.size());

    std::string bestName = m_centroidNames[0];
    double bestScore = m_centroids[0](m_normalizer(face.descriptor));

    for (size_t i=1; i<m_centroids.size(); i++) {
        const double score = m_centroids[i](m_normalizer(face.descriptor));

        // Lower is better
        if (score > bestScore) {
            continue;
        }
        bestScore = score;
        bestName = m_centroidNames[i];
    }

    if (score) {
        *score = bestScore;
    }

    return QString::fromStdString(bestName);
}

QString Database::findSlow(const Face &face, double *score)
{
    double bestScore = std::numeric_limits<double>::max();
    std::string bestName;

    for (size_t i=0; i<m_allDescriptors.size(); i++) {
        std::vector<double> scores;
        std::transform(m_allDescriptors[i].begin(), m_allDescriptors[i].end(), std::back_inserter(scores),
            [&](const dlib::matrix<float,128,1> &sample) {
                return dlib::length(face.descriptor - sample);
            }
        );
        if (scores.empty()) {
            qWarning() << "No scores for" << QString::fromStdString(m_allNames[i]);
            continue;
        }

        std::vector<double>::iterator left = scores.begin();
        std::vector<double>::iterator right = scores.end();
        right = std::unique(left, right); // eliminate duplicate outliers

        // If there's a lot for this person, try to make it a bit more "fair" by only considering the most representative
        if (scores.size() > 15) {
            std::vector<double>::iterator mid = left + scores.size() / 2;
            const size_t quarter = scores.size() / 4;
            left += quarter;
            right = mid + quarter;

            std::nth_element(scores.begin(), left, scores.end());
            std::nth_element(left + 1, mid, scores.end());
            std::nth_element(mid + 1, right, scores.end());
        }

        const double score = std::sqrt( std::inner_product(left, right, left, 0.) / std::distance(left, right));
        // Lower is better
        if (score > bestScore) {
            continue;
        }

        bestScore = score;
        bestName = m_allNames[i];
    }

    if (score) {
        *score = bestScore;
    }

    return QString::fromStdString(bestName);
}

bool Database::save(const QString &path)
{
    try {
        dlib::serialize(path.toStdString())
            << m_allDescriptors << m_allNames
            << m_centroids << m_centroidNames
            << m_allImageIds;
    } catch (const std::exception &e) {
        qWarning() << e.what();
        return false;
    }
    return true;
}

bool Database::load(const QString &path)
{
    try {
        dlib::deserialize(path.toStdString())
            >> m_allDescriptors >> m_allNames
            >> m_centroids >> m_centroidNames
            >> m_allImageIds;
        m_centroidsOutdated = false;
    } catch (const std::exception &e) {
        qWarning() << e.what();
        return false;
    }
    return true;
}

void Database::groupUnknownFaces(QVector<Face> *faces)
{
    std::vector<dlib::sample_pair> connections;
    for (int i = 0; i < faces->size(); ++i) {
        for (int j = i; j < faces->size(); ++j) {
            // The network is trained on 0.6, but 0.5 seems to give better results overall.
            // Misses some faces, though, but I think this is better for our usage
            if (length((*faces)[i].descriptor - (*faces)[j].descriptor) < 0.5) {
                connections.push_back(dlib::sample_pair(i,j));
            }
        }
    }

    std::vector<uint64_t> clusterIds;
    const uint64_t clustersCount = dlib::chinese_whispers(connections, clusterIds);

    for (size_t cluster = 0; cluster < clustersCount; ++cluster) {
        const QString name = QObject::tr("Unknown %1").arg(cluster);
        for (size_t j = 0; j < clusterIds.size(); ++j) {
            if (cluster == clusterIds[j]) {
                (*faces)[j].name = name;
            }
        }
    }
}

void Database::addFace(const Face &face, const QString &qname)
{
    const std::string name = qname.toStdString();
    int insertPos = m_allNames.size();

    std::vector<std::string>::iterator it = std::find(m_allNames.begin(), m_allNames.end(), name);
    if (it == m_allNames.end()) {
        m_allNames.push_back(name);
    } else {
        insertPos = std::distance(m_allNames.begin(), it);
    }

    m_allDescriptors[insertPos].push_back(face.descriptor);
    m_allImageIds[insertPos].push_back(face.imageId.toStdString());

    m_centroidsOutdated = true;
}

void Database::updateCache()
{
    m_normalizer = dlib::vector_normalizer<dlib::matrix<float,128,1>>();
    for (const std::vector<dlib::matrix<float,128,1>> &descriptors : m_allDescriptors) {
        m_normalizer.train(descriptors);
    }
    assert(m_allDescriptors.size() == m_allNames.size());

    m_centroids.clear();
    m_centroidNames.clear();

    for (size_t i=0; i<m_allDescriptors.size(); i++) {
        dlib::kcentroid<kernel_type> centroid(kernel_type(0.001), 0.000001, 15);
        for (const dlib::matrix<float,128,1> &descriptor : m_allDescriptors[i]) {
            centroid.train(descriptor);
        }
        m_centroids.push_back(std::move(centroid));
        m_centroidNames.push_back(m_allNames[i]);
    }

    m_centroidsOutdated = false;
}
