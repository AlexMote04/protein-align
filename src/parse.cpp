#include "parse.h"
#include "params.h"
#include <fstream>
#include <stdexcept>
#include <algorithm>

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

// This function reads a single sequence from a FASTA file, maps each amino acid to it's internal
// unsigned char representation and stores the converted vector into query_seq
void parseQuery(std::vector<unsigned char> &query_seq, const std::string &query_path)
{
    // Read exactly 1 sequence
    auto seqs = readFastaSeqs(query_path, 1);

    if (seqs[0].size() > MAX_QUERY_LEN)
        throw std::runtime_error("query sequence must be shorter than " + std::to_string(MAX_QUERY_LEN + 1));

    query_seq.clear();
    for (unsigned char c : seqs[0])
    {
        query_seq.push_back((c < 128) ? amino_to_uchar[c] : 23);
    }
}

// This function reads a single sequence from a FASTA file into query_seq
void parseQueryParasail(std::string &query_seq, const std::string &query_path)
{
    auto seqs = readFastaSeqs(query_path, 1);
    query_seq = seqs[0];
}

// ------------------------------------------------------------------
// DATABASE PARSERS
// ------------------------------------------------------------------

// This function reads and converts up to num_seqs sequences from a FASTA file and stores the converted characters sequentially in db_residues
// db_offsets stores the start index of each sequence in the vector
void parseDB(std::vector<unsigned char> &db_residues, std::vector<int> &db_offsets, const std::string &db_path, int num_seqs)
{
    if (num_seqs < 1 || num_seqs > MAX_NUM_SEQS)
        throw std::runtime_error("db_size out of bounds");

    auto seqs = readFastaSeqs(db_path, num_seqs);

    if (seqs.size() < num_seqs)
        throw std::runtime_error("Tried parsing " + std::to_string(num_seqs) + " sequences but only found " + std::to_string(seqs.size()));

    db_residues.clear();
    db_offsets.clear();

    int offset = 0;
    for (const auto &seq : seqs)
    {
        db_offsets.push_back(offset);
        for (unsigned char c : seq)
        {
            db_residues.push_back((c < 128) ? amino_to_uchar[c] : 23);
            offset++;
        }
    }
    db_offsets.push_back(offset); // Extra offset for final sequence length
}

// This function reads and converts up to num_seqs sequences from a FASTA file and stores the characters in batches of 32 characters (SoA)
// db_offsets stores the ORIGINAL start index of each sequence in the vector (not in SoA form, so is only used for getting the lengths of sequences)
void parseDBSoA(std::vector<unsigned char> &db_residues, std::vector<int> &db_offsets, const std::string &db_path, int num_seqs)
{
    if (num_seqs < 1 || num_seqs > MAX_NUM_SEQS)
        throw std::runtime_error("db_size out of bounds");

    auto seqs = readFastaSeqs(db_path, num_seqs);

    if (seqs.size() < num_seqs)
        throw std::runtime_error("Tried parsing " + std::to_string(num_seqs) + " sequences but only found " + std::to_string(seqs.size()));

    db_residues.clear();
    db_offsets.clear();

    // Populate offsets (unpadded)
    int unpadded_offset = 0;
    for (const auto &seq : seqs)
    {
        db_offsets.push_back(unpadded_offset);
        unpadded_offset += seq.size();
    }
    db_offsets.push_back(unpadded_offset);

    // Interleave and format as Structure of Arrays (SoA)
    for (size_t b = 0; b < seqs.size(); b += 32) // Loop through each batch of 32
    {
        // Last batch could be less that 32 so take min
        size_t batch_end = std::min(seqs.size(), b + 32);

        // Calculate longest sequence in batch
        size_t max_len = 0;
        for (size_t k = b; k < batch_end; k++)
        {
            max_len = std::max(seqs[k].size(), max_len);
        }

        // Need max_len structures of 32 characters
        for (size_t i = 0; i < max_len; i++)
        {
            for (size_t j = 0; j < 32; j++)
            {
                size_t seq_idx = b + j;
                if (seq_idx < seqs.size() && i < seqs[seq_idx].size())
                {
                    // Add character i of sequence seq_idx to structure
                    unsigned char c = seqs[seq_idx][i];
                    db_residues.push_back((c < 128) ? amino_to_uchar[c] : 23);
                }
                else
                {
                    // Either this sequence is shorter than max sequence in batch
                    // or this sequence is outside the batch size (final batch)
                    db_residues.push_back(23); // Pad with unknown
                }
            }
        }
    }
}

// This function reads and converts up to num_seqs sequences from a FASTA file and stores the raw characters as a flat string in db_ascii
// db_offsets stores the start index of each sequence in the string
void parseDBParasail(std::vector<std::string> &db_ascii, const std::string &db_path, int num_seqs)
{
    if (num_seqs < 1)
        throw std::runtime_error("db_size must be at least 1");

    db_ascii = readFastaSeqs(db_path, num_seqs);

    if (db_ascii.size() < num_seqs)
        throw std::runtime_error("Tried parsing " + std::to_string(num_seqs) + " sequences but only found " + std::to_string(db_ascii.size()));
}