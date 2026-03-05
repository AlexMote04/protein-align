#include "nw_parasail.h"

void nwParasail(std::vector<int> &cpu_scores, const std::string &query, const std::vector<std::string> &db_ascii)
{
  // 1. Create the SIMD profile ONCE outside the loop.
  // '_sat' automatically chooses 8-bit or 16-bit math depending on score sizes to prevent overflow.
  parasail_profile_t *profile = parasail_profile_create_sat(query.c_str(), query.length(), &parasail_blosum62);

  // 2. Loop through the database
  for (size_t i = 0; i < db_ascii.size(); i++)
  {
    // Run the Needleman-Wunsch 'scan' algorithm (the fastest SIMD approach for NW)
    parasail_result_t *result = parasail_nw_scan_profile_sat(
        profile,
        db_ascii[i].c_str(),
        db_ascii[i].length(),
        OPEN,
        EXTEND);

    cpu_scores[i] = result->score;

    // Free the result object for this specific alignment
    parasail_result_free(result);
  }

  // 3. Free the profile once you are completely done
  parasail_profile_free(profile);
}