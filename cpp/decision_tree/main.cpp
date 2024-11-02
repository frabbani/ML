#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <unordered_set>
#include <memory>
#include <iostream>
#include <optional>
#include <variant>
#include <algorithm>
#include <sstream>

#include "binner.h"
#include "datagen.h"

#define CLAMP( v, a, b ){ v = v < (a) ? (a) : v > (b) ? (b) : v; }

using namespace std;

struct Gini {
  string value = "";
  int ys = 0;
  int ns = 0;
  int total = 0;
  double impurity = 0.0;
};

vector<string> features;
map<string, vector<Gini>> featureGinis;
map<string, set<string>> featureValues;
vector<string> sortedFeatures;
map<string, map<string, Gini>> featureValueGini;

map<string, double> featureImpurity;
map<string, vector<string>> dataset;
vector<unordered_map<string, string>> datapoints;
std::map<string, Binner> featureBins;

struct Node {
  string feature = "";
  unordered_map<string, Node*> children;
  struct Label {
    int no = 0;
    int yes = 0;
    string value() {
      return no > yes ? "No" : "Yes";
    }
  };
  optional<Label> label;
  bool isLeaf() {
    return label.has_value();
  }
};

Node *tree = nullptr;

Node* createNode(int index) {
  int last = int(sortedFeatures.size()) - 1;
  if (index > last)
    return nullptr;

  Node *node = new Node;
  node->feature = sortedFeatures[index];
  if (index == last) {
    node->label = Node::Label();
    return node;
  }
  auto &values = featureValues[node->feature];
  for (auto value : values) {
    auto &gini = featureValueGini[node->feature][value];
    if (gini.impurity < 0.4)
      node->children[value] = createNode(index + 1);  // good question, ask another one!
    else {
      // question's response will be mixed, this line of questioning is a fail so we rely on odds
      node->children[value] = new Node;
      node->children[value]->feature = node->feature;
      node->children[value]->label = Node::Label();
    }
  }
  return node;
}


void walkNodes(Node *node, unordered_map<string, string> &datapoint,
                vector<string> &path) {
  auto value = datapoint[node->feature];

  // a high count of either "yes" or "no" indicates good odds of picking the right one
  if (node->isLeaf()) {
    if (datapoint["Label"] == "Yes")
      node->label->yes++;
    else
      node->label->no++;
    path.push_back('(' + node->label->value() + ')');
    return;
  }

  path.push_back(node->feature + "=" + value);
  for (auto& [v, n] : node->children) {
    if (v == value && n) {
      path.push_back("->");
      walkNodes(n, datapoint, path);
      return;
    }
  }
}

string classify(Node *node, unordered_map<string, string> &datapoint,
                 vector<string> &path) {
  auto value = datapoint[node->feature];

  if (node->isLeaf()) {
    path.push_back('(' + node->label->value() + ')');
    return node->label->yes >= node->label->no ? "Yes" : "No";
  }

  path.push_back(node->feature + "=" + value);
  for (auto& [childValue, child] : node->children) {
    if (child && childValue == value) {
      path.push_back("->");
      return classify(child, datapoint, path);
    }
  }
  return "???";
}

void printNodes(Node *node, int indent, FILE *fp) {
  if (!node)
    return;  // Base case: if the node is null, do nothing

  auto spaces = [](int count) {
    char blanks[32] = { '\0' };
    for (int i = 0; i < count; i++)
      blanks[i] = ' ';
    return string(blanks);
  };

  if (node->isLeaf()) {
    return;
  }

  // Print the feature of the current node
  fprintf(fp, "%s%s\n", spaces(indent).c_str(), node->feature.c_str());
  indent += 2;
  // Iterate through children and print each one
  for (const auto &child : node->children) {
    fprintf(fp, "%s%s:\n", spaces(indent).c_str(), child.first.c_str());  // Print the edge
    printNodes(child.second, indent + 2, fp);  // Recursively print the child node with increased indentation
  }
}

void loadFile(string_view name = "dataset5.csv") {
  FILE *fp = fopen(name.data(), "r");
  char line[1024];
  char *token = nullptr;
  fgets(line, sizeof(line), fp);
  token = strtok(line, ",\n\r\t");
  while (token) {
    features.push_back(string(token));
    token = strtok(nullptr, ",\n\r\t");
  }

  while (fgets(line, sizeof(line), fp)) {
    token = strtok(line, ",\n\r");
    for (auto feature : features) {
      dataset[feature].push_back(string(token));
      token = strtok(nullptr, ",\n\r");
    }
  }

  auto is_numeric = [=](const vector<string> &values) {
    return std::all_of(values.begin(), values.end(), [](const string &value) {
      for (auto c : value)
        if (!isdigit(c))
          return false;
      return true;
    });
  };

  for (auto feature : features) {
    if (!is_numeric(dataset[feature]))
      continue;
    printf("feature '%s' is continuous\n", feature.c_str());
    vector<double> values;
    for (auto value : dataset[feature]) {
      double number;
      stringstream ss;
      ss << value;
      ss >> number;
      values.push_back(number);
    }
    auto &binner = featureBins[feature];
    binner.createBins(values, feature);
    int i = 0;
    for (auto &value : dataset[feature])
      value = binner.bin(values[i++]);

  }

  int n = 0;
  for (auto& [f, i] : dataset) {
    n = int(i.size());
    break;
  }

  for (int i = 0; i < n; i++) {
    unordered_map<string, string> row;
    for (auto feature : features)
      row[feature] = dataset[feature][i];
    datapoints.push_back(std::move(row));
  }

//  for (auto& [f, i] : dataset) {
//    printf("%d rows\n", int(i.size()));
//  }
//  for (auto feature : features) {
//    printf("feature: '%s'\n", feature.c_str());
//    for (auto instance : dataset[feature])
//      printf(" + '%s'\n", instance.c_str());
//  }
}

Gini impurity(string_view feature, string_view value) {
  const auto &instances = dataset[string(feature)];
  const auto &labelInstances = dataset["Label"];

  Gini gini;
  gini.value = string(value);
  for (int i = 0; i < int(instances.size()); i++) {
    if (instances[i] != value)
      continue;
    gini.total++;
    if (labelInstances[i] == "Yes")
      gini.ys++;
    else
      gini.ns++;
  }
  if (0 == gini.total) {
    gini.impurity = 1.0;
    return gini;
  }

  double p = double(gini.ys) / double(gini.total);
  double p2 = double(gini.ns) / double(gini.total);

  gini.impurity = 1.0 - (p * p + p2 * p2);
  return gini;
}

void load() {
  printf("***************\n");
  loadFile("dataset5.csv");
  printf("***************\n");

  auto sum = [](vector<Gini> &ginis) {
    int total = 0;
    for (auto g : ginis)
      total += g.total;
    double purity = 0.0;
    for (auto g : ginis)
      purity += double(g.total) / double(total) * g.impurity;
    return purity;
  };

  for (auto &feature : features) {
    set<string> unique(dataset[feature].begin(), dataset[feature].end());
    featureValues[feature] = std::move(unique);
    printf("feature: '%s'\n", feature.c_str());
  }

  for (auto &feature : features) {
    auto &ginis = featureGinis[feature];
    ginis.clear();
    if (feature == "Label") {
      featureImpurity[feature] = 1.0;
      continue;
    }
    for (auto &value : featureValues[feature]) {
      auto gini = impurity(feature, value);
      featureValueGini[feature][value] = gini;
      ginis.push_back(std::move(gini));
    }
    featureImpurity[feature] = sum(ginis);
  }

  printf("***************\n");
  for (auto &feature : features) {
    auto &ginis = featureGinis[feature];
    printf("'%s': %lf\n", feature.c_str(), featureImpurity[feature]);
    int n = int(ginis.size());
    for (auto gini : ginis) {
      printf(" + '%s' %.3lf (%d of %d: %lf)\n", gini.value.c_str(),
             gini.impurity, gini.total, n, double(gini.total) / double(n));
    }
  }
}

void build() {
  map<double, vector<string>> featureMap;
  for (auto &feature : features) {
    featureMap[featureImpurity[feature]].push_back(feature);
  }
  sortedFeatures.clear();
  for (auto& [purity, features] : featureMap) {
    if (0.0 == purity) {
      sortedFeatures.push_back(features[0]);
      continue;
    }
    for (auto feature : features)
      sortedFeatures.push_back(feature);
  }
  printf("***************\n");
  for (auto feature : sortedFeatures) {
    printf("feature '%s' gini %lf\n", feature.c_str(),
           featureImpurity[feature]);
  }
  tree = createNode(0);
//printf("***************\n");
//printNode(tree);

}

void train() {

//  for (auto &row : datapoints) {
//    for (auto feature : features)
//      printf("%s:%s", feature.c_str(), row[feature].c_str());
//    printf("\n");
//  }
  printf("***************\n");
  vector<string> path;
  for (auto &row : datapoints) {
    path.clear();
    walkNodes(tree, row, path);
    for (auto step : path)
      printf("%s ", step.c_str());
    printf("| (%s)\n", row["Label"].c_str());
  }

  printf("***************\n");
  //printNode(tree);
}

int test() {
  auto sample = testSample();
  unordered_map<string, string> datapoint;
  datapoint["Age"] = featureBins["Age"].bin(get<0>(sample));
  datapoint["Income"] = featureBins["Income"].bin(get<1>(sample));
  datapoint["Education"] = get<2>(sample);
  datapoint["Marital Status"] = get<3>(sample);
  datapoint["Occupation"] = get<4>(sample);
  datapoint["Label"] = get<5>(sample);
  vector<string> path;
  auto label = classify(tree, datapoint, path);
  for (auto step : path)
    printf("%s ", step.c_str());
  bool correct = label == datapoint["Label"];
  printf("%s\n", correct ? "" : " - X");
  return correct;

}

int main() {
  printf("hello world!\n");
  srand(123);

  writeData("dataset5.csv", 200);

  load();
  build();
  train();

  FILE *fp = fopen("tree.txt", "w");
  if (fp) {
    printNodes(tree, 0, fp);
    fclose(fp);
  }

  int numTests = 100;
  int numCorrects = 0;
  for (int i = 0; i < numTests; i++)
    numCorrects += test();
  printf("accuracy: %d of %d (%.4lf)\n", numCorrects, numTests,
         double(numCorrects) / double(numTests));

  printf("goodbye!\n");
  return 0;
}

