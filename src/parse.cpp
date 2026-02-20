#include "parse.h"

unsigned char amino_to_uchar[128];

struct TableInitializer {
    TableInitializer() {
        initConversionTable();
    }
};

static TableInitializer auto_init;

void initConversionTable(){
  // Initialize everything to unknown (24)
  for(int i = 0; i < 128; i++){
    amino_to_uchar[i] = 22;
  }

  // Standard Amino Acids
  amino_to_uchar['A'] = amino_to_uchar['a'] = 0;
  amino_to_uchar['R'] = amino_to_uchar['r'] = 1;
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

  // Special Codes
  amino_to_uchar['B'] = amino_to_uchar['b'] = 20; // D or N
  amino_to_uchar['Z'] = amino_to_uchar['z'] = 21; // E or Q
  amino_to_uchar['X'] = amino_to_uchar['x'] = 22; // Any
  amino_to_uchar['*'] = 23;
}

int parseQuery(std::vector<unsigned char> &query, std::string query_path){
  // Open query file
  std::ifstream file(query_path);
  if(!file) throw std::runtime_error("Could not open file: " + query_path);

  // Read query sequence into string
  std::string query_string;
  std::string line;

  // Parse header line
  if(!std::getline(file, line) || line.empty() || line[0] != '>'){
    throw std::runtime_error("Invalid FASTA: Missing header line");
  }

  while(std::getline(file, line)){
    if(line.empty()) continue;
    
    // Clean up potential Windows charriage returns
    if(line.back() == '\r') line.pop_back();

    // Map string to unsigned chars and add to vector
    for(unsigned char c : line){
      if (c < 128) {
        query.push_back(amino_to_uchar[c]);
      } else {
        query.push_back(22); // Defult to unknown for weird characters
      }
    }
  }

  if(query.empty()){
    throw std::runtime_error("Invalid query file: No query sequence found");
  }

  // Convert query string into vector of unsigned chars
  return 0;
}

int parseDB(std::vector<unsigned char> &db, std::vector<int> &offsets, std::string db_path, int db_size){
  if(db_size < 1 || db_size > MAX_DB_LENGTH) throw std::runtime_error("db_size must be between 1 and " + MAX_DB_LENGTH);

  // Open database file
  std::ifstream file(db_path);
  if(!file) throw std::runtime_error("Could not open file: " + db_path);

  // Read sequence into string
  std::string line;
  int offset = 0;

  // Parse header line
  if(!std::getline(file, line) || line.empty() || line[0] != '>'){
    throw std::runtime_error("Invalid FASTA: Missing header line");
  }

  // First header seen, so start of sequence
  offsets.push_back(offset);
  bool in_seq = false;

  while(std::getline(file, line)){
    if(line.empty()) continue;

    if(line[0] == '>'){
      if(!in_seq) throw std::runtime_error("Invalid Fasta: Header with no sequence");

      // Check if correct number of sequences have been parsed
      if(offsets.size() == db_size) return 0; 

      // Header means start of sequence
      in_seq = false;
      offsets.push_back(offset);
    } else {
      in_seq = true;
      // Map sequence to NCBI numbers
      for (unsigned char c : line){
        if (c < 128){
          db.push_back(amino_to_uchar[c]);
        } else {
          db.push_back(24);
        }
        offset++; // Keep track for current offset  
      }
    }
  }

  if(db.empty()) throw std::runtime_error("Invalid database file: No sequences found");
  if(offsets.size() < db_size) throw std::runtime_error("Tried parsing " + std::to_string(db_size) + " sequences but only found " + std::to_string(offsets.size()) + " in database file");
  
  return 0;
}