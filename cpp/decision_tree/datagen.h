#pragma once

#include <random>
#include <string>
#include <string_view>
#include <vector>

using namespace std;

using Sample = tuple<int, int, string, string, string, string>;

const vector<string> educationLevels = { "High School", "Bachelors", "Masters",
    "PhD" };
const vector<string> maritalStatuses = { "Single", "Married", "Divorced",
    "Widowed" };
const vector<string> occupations = { "Engineer", "Doctor", "Teacher", "Artist",
    "Nurse", "Lawyer", "Sales", "Manager" };


const int minIncome = 50000 / 100;
const int maxIncome = 150000 / 100;
const int minAge = 18;
const int maxAge = 65;

int randomValue(int min, int max) {
  static std::random_device rd;
  static std::mt19937 gen(rd());
  //uniform_real_distribution<> dis(min, max);
  uniform_int_distribution<> dis(min, max);
  return dis(gen);
}

// Function to generate random categorical values
string randomCategory(const vector<string> &categories) {
  static random_device rd;
  static mt19937 gen(rd());
  uniform_int_distribution<> dis(0, categories.size() - 1);
  return categories[dis(gen)];
}

string classify(int age, int income, string_view education,
                string_view maritalStatus, string_view occupation) {
  string label = "No";  // Default label
  if (income > 120000
      && (education == "Bachelors" || education == "Masters"
          || education == "PhD")) {
    label = "Yes";  // High income with higher education
  } else if (income >= 80000 && income <= 120000
      && (occupation == "Doctor" || occupation == "Engineer")) {
    label = "Yes";  // High earning professions
  } else if (age > 50 && income < 70000) {
    label = "No";  // Older age with low income
  } else if (income < 50000) {
    label = "No";  // Low income threshold
  } else if (education == "Masters" || education == "PhD") {
    label = "Yes";  // High education alone
  }
  return label;
}

Sample testSample() {
  int age = randomValue(minAge, maxAge);
  int income = randomValue(minIncome, maxIncome) * 100;
  string education = randomCategory(educationLevels);
  string maritalStatus = randomCategory(maritalStatuses);
  string occupation = randomCategory(occupations);
  string label = classify(age, income, education, maritalStatus, occupation);
  return Sample(age, income, education, maritalStatus, occupation, label);
}

void writeData(string_view fileName, int numSamples = 25) {
  vector<Sample> samples;
  for (int i = 0; i < numSamples; ++i) {
    auto sample = testSample();
    samples.push_back(std::move(sample));
  }
  FILE *fp = fopen(fileName.data(), "w");
  fprintf(fp, "Age,Income,Education,Marital Status,Occupation,Label\n");
  for (auto sample : samples) {
    fprintf(fp, "%d,%d,%s,%s,%s,%s\n", get<0>(sample), get<1>(sample),
            get<2>(sample).c_str(), get<3>(sample).c_str(),
            get<4>(sample).c_str(), get<5>(sample).c_str());
  }
  fclose(fp);
}

