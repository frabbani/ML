#pragma once

#include <cmath>
#include <vector>
#include <algorithm>
#include <map>

using namespace std;

struct Binner {

  int numClusters = 0;
  vector<double> centroids;
  vector<string> values;
  vector<vector<double>> clusters;

  void createBins(std::vector<double> &data, string tag) {
    numClusters = int(sqrt(double(data.size())));
    numClusters = 0 == numClusters ? 1 : numClusters;

    for (int i = 0; i < numClusters; i++) {
      int index =
          (double(i) / double(numClusters - 1) * double(data.size() - 1));
      centroids.push_back(data[index]);
    }
    clusters = vector<vector<double>>(numClusters);

    auto get_cluster = [&](double v) {
      int cluster = 0;
      double dist = fabs(centroids[cluster] - v);
      for (int i = 1; i < int(centroids.size()); i++) {
        double dist2 = fabs(centroids[i] - v);
        if (dist2 < dist) {
          cluster = i;
          dist = dist2;
        }
      }
      return cluster;
    };

    auto get_centroid = [=](std::vector<double> &values) {
      if (values.empty())
        return 0.0;
      double value = 0.0;
      for (auto v : values)
        value += v;
      return value / double(values.size());
    };

    auto learn = [&]() {
      for (auto &cluster : clusters)
        cluster.clear();

      for (auto value : data) {
        auto cluster = get_cluster(value);
        clusters[cluster].push_back(value);
      }
      for (size_t i = 0; i < clusters.size(); i++) {
        centroids[i] = get_centroid(clusters[i]);
      }
    };

    for (int i = 0; i < 50; i++)
      learn();

    values.clear();
    for (int i = 0; i < (int) centroids.size(); i++) {
      values.push_back(tag + "Range" + to_string(i));
    }
  }

  string bin(double value) {
    int cluster = 0;
    double dist = fabs(centroids[cluster] - value);
    for (int i = 1; i < int(centroids.size()); i++) {
      double dist2 = fabs(centroids[i] - value);
      if (dist2 < dist) {
        cluster = i;
        dist = dist2;
      }
    }
    return values[cluster];
  }

};
