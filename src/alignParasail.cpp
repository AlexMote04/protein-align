#include "alignParasail.h"

void alignParasail(const int algorithm, std::vector<int> &scores, const std::string &query, const std::vector<std::string> &db_ascii)
{
  // Create the SIMD profile ONCE outside the loop.
  // '_sat' automatically chooses 8-bit or 16-bit math depending on score sizes to prevent overflow.
  parasail_profile_t *profile = parasail_profile_create_sat(query.c_str(), query.length(), &parasail_blosum62);

  if (algorithm == 0)
  {
    // Loop through the database
    for (size_t i = 0; i < db_ascii.size(); i++)
    {
      // Run the Needleman-Wunsch 'scan' algorithm (the fastest SIMD approach for NW)
      parasail_result_t *result = parasail_nw_scan_profile_sat(
          profile,
          db_ascii[i].c_str(),
          db_ascii[i].length(),
          OPEN,
          EXTEND);

      scores[i] = result->score;

      // Free the result object for this specific alignment
      parasail_result_free(result);
    }
  }
  else
  {
    for (size_t i = 0; i < db_ascii.size(); i++)
    {
      // Run the Smith-Waterman'scan' algorithm
      parasail_result_t *result = parasail_sw_scan_profile_sat(
          profile,
          db_ascii[i].c_str(),
          db_ascii[i].length(),
          OPEN,
          EXTEND);

      scores[i] = result->score;

      // Free the result object for this specific alignment
      parasail_result_free(result);
    }
  }

  // Free the profile once you are completely done
  parasail_profile_free(profile);
}