#include "parse.h"

void initConversionTable(){
  // Initialize everything to unknown (23)
  for(int i = 0; i < 128; i++){
    amino_to_uchar[i] = 23;
  }

  // Set specific mappings (NCBI Order)
  // Handle both Upper and Lowercase
  amino_to_uchar['R'] = amino_to_uchar['r'] = 1;
  amino_to_uchar['A'] = amino_to_uchar['a'] = 0;
  amino_to_uchar['N'] = amino_to_uchar['n'] = 2;
  amino_to_uchar['D'] = amino_to_uchar['d'] = 3;
  amino_to_uchar['C'] = amino_to_uchar['c'] = 4;
  amino_to_uchar['Q'] = amino_to_uchar['q'] = 5;
  amino_to_uchar['E'] = amino_to_uchar['e'] = 6;
  amino_to_uchar['G'] = amino_to_uchar['g'] = 7;
  amino_to_uchar['H'] = amino_to_uchar['h'] = 8;
  amino_to_uchar['I'] = amino_to_uchar['i'] = 9;
  amino_to_uchar['L'] = amino_to_uchar['l'] = 10;
  amino_to_uchar['K'] = amino_to_uchar['k'] = 11;
  amino_to_uchar['M'] = amino_to_uchar['m'] = 12;
  amino_to_uchar['F'] = amino_to_uchar['f'] = 13;
  amino_to_uchar['P'] = amino_to_uchar['p'] = 14;
  amino_to_uchar['S'] = amino_to_uchar['s'] = 15;
  amino_to_uchar['T'] = amino_to_uchar['t'] = 16;
  amino_to_uchar['W'] = amino_to_uchar['w'] = 17;
  amino_to_uchar['Y'] = amino_to_uchar['y'] = 18;
  amino_to_uchar['V'] = amino_to_uchar['v'] = 19;
  amino_to_uchar['B'] = amino_to_uchar['b'] = 20;
  amino_to_uchar['Z'] = amino_to_uchar['z'] = 21;
  amino_to_uchar['X'] = amino_to_uchar['x'] = 22;
  amino_to_uchar['*'] = 23;
}

int parseQuery(std::vector<int> query, std::string query_path){
  // Open query file
  std::ifstream query_file(query_path);

  // Read query sequence into string
  std::string query_string;
  std::string line;
  while(std::getline(query_file, line)){
    if(line.length() > 0 && line[0] != '>'){
      // Only process line if not empty and not description
      query_string += line;
    }
  }

  // Convert query string into vector of unsigned chars
  for (int i = 0; i < query_string.length(); i++){
    query.push_back(amino_to_uchar[query_string[i]]);
  }
  return 0;
}
int parseDB(std::vector<unsigned char> db, std::vector<int> offsets, std::string db_path, int db_size){

}