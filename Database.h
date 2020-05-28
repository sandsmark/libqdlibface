#pragma once

#include <QString>

#include <dlib/svm/kcentroid.h>
#include <vector>
#include <string>

struct Face;

class Database
{
public:
    void addFace(const QString &name, const Face &face);

    // Is a bit faster, but uses centroids that approximate matching
    QString findFast(const Face &face, double *score = nullptr);

    // More brute force search through all face descriptors
    QString findSlow(const Face &face, double *score = nullptr);

    bool save(const QString &path);
    bool load(const QString &path);

    /// Updates the centroids
    void updateCache();

private:
    using kernel_type = dlib::radial_basis_kernel<dlib::matrix<float,128,1>>;
    std::vector<dlib::kcentroid<kernel_type>> m_centroids;
    dlib::vector_normalizer<dlib::matrix<float,128,1>> m_normalizer;
    std::vector<std::string> m_centroidNames;

    std::vector<std::vector<dlib::matrix<float,128,1>>> m_allDescriptors;
    std::vector<std::string> m_allNames;

    bool m_centroidsOutdated = true;
};
