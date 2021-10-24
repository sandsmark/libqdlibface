#pragma once

#include <QString>

#include <dlib/svm/kcentroid.h>
#include <vector>
#include <string>

struct Face;

class Database
{
public:
    void addFace(const Face &face, const QString &name = QString());

    // Is a bit faster, but uses centroids that approximate matching
    QString findFast(const Face &face, double *score = nullptr);

    // More brute force search through all face descriptors
    QString findSlow(const Face &face, double *score = nullptr);

    bool save(const QString &path);
    bool load(const QString &path);

    /// Updates the centroids
    void updateCache();

    /// Automatically tries to group similar faces (by clustering) and assigns
    /// "Unknown #" names to all faces passed
    static void groupUnknownFaces(QVector<Face> *faces);

    void rename(const QString &oldName, const QString &newName);
    void deleteAll(const QString &name);
    QStringList allNames() const;

    QVector<Face> facesForName(const QString &name) const;

private:
    using kernel_type = dlib::radial_basis_kernel<dlib::matrix<float,128,1>>;
    std::vector<dlib::kcentroid<kernel_type>> m_centroids;
    dlib::vector_normalizer<dlib::matrix<float,128,1>> m_normalizer;
    std::vector<std::string> m_centroidNames;

    std::vector<std::vector<Face>> m_allFaces;
    std::vector<std::string> m_allNames;

    std::vector<Face> m_unknownFaces;

    bool m_centroidsOutdated = true;
};
