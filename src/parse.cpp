#include "parse.h"
#include "params.h"
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <string_view>
#include <limits>

unsigned char amino_to_uchar[128];

struct TableInitializer
{
    TableInitializer()
    {
        initConversionTable();
    }
};

static TableInitializer auto_init;

void initConversionTable()
{
    // Initialize everything to unknown (23)
    for (int i = 0; i < 128; i++)
    {
        amino_to_uchar[i] = 23;
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

// This function reads up to max_seqs from a FASTA file and returns them as an array of strings.
// Passing max_seqs = 1 acts exactly like parsing a query.
static std::vector<std::string> readFastaSeqs(const std::string &path, int max_seqs)
{
    std::ifstream file(path);
    if (!file)
        throw std::runtime_error("Could not open file: " + path);

    std::vector<std::string> sequences;
    std::string line;
    std::string current_seq = "";
    bool in_seq = false;

    if (!std::getline(file, line) || line.empty() || line[0] != '>')
        throw std::runtime_error("Invalid FASTA: Missing header line in " + path);

    while (std::getline(file, line))
    {
        if (line.empty())
            continue;

        if (line.back() == '\r')
            line.pop_back();

        if (line[0] == '>')
        {
            if (!in_seq)
                throw std::runtime_error("Invalid FASTA: Header with no sequence in " + path);

            sequences.push_back(current_seq);
            current_seq.clear();

            if (sequences.size() == max_seqs)
                break;

            in_seq = false;
        }
        else
        {
            in_seq = true;
            current_seq += line;
        }
    }

    if (in_seq && sequences.size() < max_seqs)
    {
        sequences.push_back(current_seq);
    }

    if (sequences.empty())
        throw std::runtime_error("Invalid FASTA: No sequences found in " + path);

    return sequences;
}

// ------------------------------------------------------------------
// QUERY PARSERS
// ------------------------------------------------------------------

void parseQuery(std::vector<unsigned char> &query_seq, const std::string &query_path)
{
    auto seqs = readFastaSeqs(query_path, 1);

    if (seqs[0].size() > MAX_QUERY_LEN)
        throw std::runtime_error("query sequence must be shorter than " + std::to_string(MAX_QUERY_LEN + 1));

    query_seq.clear();
    for (unsigned char c : seqs[0])
    {
        query_seq.push_back((c < 128) ? amino_to_uchar[c] : 23);
    }
}

void parseQueryParasail(std::string &query_seq, const std::string &query_path)
{
    auto seqs = readFastaSeqs(query_path, 1);
    query_seq = seqs[0];
}

// ==================================================================
// CACHING
// ==================================================================

std::vector<std::string> loadOrCacheDatabase(const std::string &fasta_path, int num_seqs)
{
    std::vector<std::string> db;

    // Try reading the binary file
    std::string cache_path = fasta_path + ".bin";
    std::ifstream bin_in(cache_path, std::ios::binary);
    if (bin_in.is_open())
    {
        int cached_num_seqs;
        bin_in.read(reinterpret_cast<char *>(&cached_num_seqs), sizeof(int));

        // Read up to what's requested, capping at what's available
        int seqs_to_load = std::min(num_seqs, cached_num_seqs);
        db.resize(seqs_to_load);

        for (int i = 0; i < seqs_to_load; i++)
        {
            int len;
            bin_in.read(reinterpret_cast<char *>(&len), sizeof(int));
            db[i].resize(len);
            bin_in.read(&db[i][0], len);
        }
        bin_in.close();
        return db;
    }

    // Cache Miss: Read the ENTIRE FASTA file directly into db
    db = readFastaSeqs(fasta_path, std::numeric_limits<int>::max());

    // Sort by sequence length
    std::sort(db.begin(), db.end(), [](const std::string &a, const std::string &b)
              { return a.length() < b.length(); });

    // Save to binary cache
    std::ofstream bin_out(cache_path, std::ios::binary);
    if (bin_out.is_open())
    {
        int total_seqs = db.size();
        bin_out.write(reinterpret_cast<char *>(&total_seqs), sizeof(int));
        for (const auto &seq : db)
        {
            int len = seq.length();
            bin_out.write(reinterpret_cast<const char *>(&len), sizeof(int));
            bin_out.write(seq.data(), len);
        }
    }

    // Slice down to requested num_seqs for this execution
    if (num_seqs < db.size())
    {
        db.resize(num_seqs);
    }

    return db;
}

// ==================================================================
// GENERATORS
// ==================================================================

void generateDBSoA(const std::vector<std::string> &sorted_db,
                   std::vector<unsigned char> &db_residues,
                   std::vector<int> &db_offsets,
                   std::vector<std::vector<unsigned char>> &massive_sequences,
                   int threshold)
{
    db_residues.clear();
    db_offsets.clear();
    massive_sequences.clear();

    auto cutoff_it = std::find_if(sorted_db.begin(), sorted_db.end(),
                                  [threshold](const std::string &seq)
                                  {
                                      return seq.size() >= threshold;
                                  });

    int num_small = std::distance(sorted_db.begin(), cutoff_it);

    int unpadded_offset = 0;
    for (int i = 0; i < num_small; i++)
    {
        db_offsets.push_back(unpadded_offset);
        unpadded_offset += sorted_db[i].size();
    }
    db_offsets.push_back(unpadded_offset);

    for (int b = 0; b < num_small; b += 32)
    {
        int batch_end = std::min(num_small, b + 32);
        size_t max_len = sorted_db[batch_end - 1].size();

        for (size_t i = 0; i < max_len; i++)
        {
            for (size_t j = 0; j < 32; j++)
            {
                int seq_idx = b + j;
                if (seq_idx < num_small && i < sorted_db[seq_idx].size())
                {
                    unsigned char c = sorted_db[seq_idx][i];
                    db_residues.push_back((c < 128) ? amino_to_uchar[c] : 23);
                }
                else
                {
                    db_residues.push_back(23);
                }
            }
        }
    }

    for (auto it = cutoff_it; it != sorted_db.end(); ++it)
    {
        std::vector<unsigned char> converted_seq;
        converted_seq.reserve(it->size());
        for (char c : *it)
        {
            converted_seq.push_back((c < 128) ? amino_to_uchar[static_cast<unsigned char>(c)] : 23);
        }
        massive_sequences.push_back(converted_seq);
    }
}

void generateDBParasail(const std::vector<std::string> &sorted_db, std::vector<std::string> &db_ascii)
{
    // Since both are now std::vector<std::string>, this is just a direct assignment
    db_ascii = sorted_db;
}