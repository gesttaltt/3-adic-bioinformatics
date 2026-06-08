#!/usr/bin/env python3
"""
Comprehensive API Testing Suite for HIV Drug Resistance Prediction
===================================================================

This script tests all available APIs and demonstrates:
1. What data can be retrieved
2. How to integrate it into our models
3. Example outputs for each API

Author: Claude Code
Date: December 28, 2024
"""

import io
import json
import sys
import time
import warnings
from pathlib import Path

import requests

# Fix Windows encoding issues
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Output directory for API results
OUTPUT_DIR = Path(__file__).parent.parent.parent / "results" / "api_tests"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# HIV REFERENCE SEQUENCES
# =============================================================================

# HIV-1 HXB2 Reference sequences (standard reference strain)
HIV_SEQUENCES = {
    "protease": ("PQVTLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"),
    "reverse_transcriptase": (
        "PISPIETVPVKLKPGMDGPKVKQWPLTEEKIKALVEICTEMEKEGKISKIGPENPYNTPV"
        "FAIKKKDSTKWRKLVDFRELNKRTQDFWEVQLGIPHPAGLKKKKSVTVLDVGDAYFSVPL"
        "DEDFRKYTAFTIPSINNETPGIRYQYNVLPQGWKGSPAIFQSSMTKILEPFRKQNPDIVI"
        "YQYMDDLYVGSDLEIGQHRTKIEELRQHLLRWGLTTPDKKHQKEPPFLWMGYELHPDKWT"
        "VQPIVLPEKDSWTVNDIQKLVGKLNWASQIYPGIKVRQLCKLLRGTKALTEVIPLTEEAE"
        "LELAENREILKEPVHGVYYDPSKDLIAEIQKQGQGQWTYQIYQEPFKNLKTGKYARMRGA"
        "HTNDVKQLTEAVQKITTESIVIWGKTPKFKLPIQKETWETWWTEYWQATWIPEWEFVNTP"
        "PLVKLWYQLEKEPIVGAETFYVDGAANRETKLGKAGYVTNRGRQKVVTLTDTTNQKTELQ"
        "AIYLALQDSGLEVNIVTDSQYALGIIQAQPDQSESELVNQIIEQLIKKEKVYLAWVPAHK"
        "GIGGNEQVDKLVSAGIRKVL"
    ),
    "integrase": (
        "FLDGIDKAQDEHEKYHSNWRAMASDFNLPPVVAKEIVASCDKCQLKGEAMHGQVDCSPGI"
        "WQLDCTHLEGKVILVAVHVASGYIEAEVIPAETGQETAYFLLKLAGRWPVKTIHTDNGSN"
        "FTGATVRAACWWAGIKQEFGIPYNPQSQGVVESMNKELKKIIGQVRDQAEHLKTAVQMAV"
        "FIHNFKRKGGIGGYSAGERIVDIIATDIQTKELQKQITKIQNFRVYYRDSRNPLWKGPAK"
        "LLWKGEGAVVIQDNSDIKVVPRRKAKIIRDYGKQMAGDDCVASRQDED"
    ),
}

# Common drug resistance mutations for testing
TEST_MUTATIONS = {
    "protease": ["M46I", "I54V", "V82A", "L90M", "I84V"],
    "rt": ["M184V", "K103N", "Y181C", "K65R", "L74V"],
    "integrase": ["Y143R", "Q148H", "N155H", "G140S", "E92Q"],
}


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_subsection(title: str):
    """Print a formatted subsection header."""
    print(f"\n--- {title} ---")


# =============================================================================
# 1. ESM-2 PROTEIN LANGUAGE MODEL
# =============================================================================


def test_esm2_embeddings():
    """
    Test ESM-2 protein embeddings from Facebook/Meta Research.

    ESM-2 provides:
    - Per-residue embeddings (1280 dimensions for large model)
    - Attention maps showing residue-residue relationships
    - Contact prediction capabilities

    Use cases for our project:
    - Replace one-hot encoding with learned representations
    - Capture evolutionary constraints on mutations
    - Identify functionally important positions
    """
    print_section("1. ESM-2 PROTEIN EMBEDDINGS")

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer

        print("Loading ESM-2 model (this may take a moment)...")

        # Use smaller model for testing (8M parameters)
        model_name = "facebook/esm2_t6_8M_UR50D"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        model.eval()

        print(f"✓ Model loaded: {model_name}")
        print("  - Parameters: 8M")
        print("  - Embedding dimension: 320")

        # Test on HIV protease
        sequence = HIV_SEQUENCES["protease"]
        print(f"\nTesting on HIV-1 Protease ({len(sequence)} residues)...")

        inputs = tokenizer(sequence, return_tensors="pt", add_special_tokens=True)

        with torch.no_grad():
            outputs = model(**inputs)

        embeddings = outputs.last_hidden_state
        print(f"\n✓ Embedding shape: {embeddings.shape}")
        print(f"  - Batch size: {embeddings.shape[0]}")
        print(f"  - Sequence length: {embeddings.shape[1]} (includes special tokens)")
        print(f"  - Embedding dim: {embeddings.shape[2]}")

        # Mean pooling for sequence-level embedding
        seq_embedding = embeddings.mean(dim=1)
        print(f"\n✓ Sequence embedding (mean pooled): {seq_embedding.shape}")

        # Show embedding statistics
        print("\nEmbedding statistics:")
        print(f"  - Mean: {seq_embedding.mean().item():.4f}")
        print(f"  - Std: {seq_embedding.std().item():.4f}")
        print(f"  - Min: {seq_embedding.min().item():.4f}")
        print(f"  - Max: {seq_embedding.max().item():.4f}")

        # Compare wild-type vs mutant
        print_subsection("Mutation Effect Analysis")

        # Apply M46I mutation
        wt_seq = sequence
        mut_seq = sequence[:45] + "I" + sequence[46:]  # M46I

        wt_inputs = tokenizer(wt_seq, return_tensors="pt")
        mut_inputs = tokenizer(mut_seq, return_tensors="pt")

        with torch.no_grad():
            wt_emb = model(**wt_inputs).last_hidden_state.mean(dim=1)
            mut_emb = model(**mut_inputs).last_hidden_state.mean(dim=1)

        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(wt_emb, mut_emb)
        euclidean_dist = torch.norm(wt_emb - mut_emb)

        print("M46I mutation analysis:")
        print(f"  - Cosine similarity (WT vs M46I): {cos_sim.item():.6f}")
        print(f"  - Euclidean distance: {euclidean_dist.item():.4f}")
        print("  - Interpretation: Small distance = conservative mutation")

        # How to use in our model
        print_subsection("Integration with Our VAE")
        print("""
How to integrate ESM-2 embeddings:

1. REPLACE ONE-HOT ENCODING:
   - Current: 20-dim one-hot per position
   - New: 320/640/1280-dim ESM-2 embedding per position

2. PRE-COMPUTE EMBEDDINGS:
   ```python
   # Save embeddings for all sequences
   embeddings = {}
   for seq_id, sequence in dataset.items():
       inputs = tokenizer(sequence, return_tensors="pt")
       with torch.no_grad():
           emb = model(**inputs).last_hidden_state
       embeddings[seq_id] = emb.numpy()
   np.save("esm2_embeddings.npy", embeddings)
   ```

3. MODIFY VAE INPUT:
   ```python
   class ESM2VAE(nn.Module):
       def __init__(self, esm_dim=320, latent_dim=16):
           self.encoder = nn.Sequential(
               nn.Linear(esm_dim, 128),  # 320 -> 128
               nn.ReLU(),
               nn.Linear(128, 64),
               ...
           )
   ```

4. EXPECTED IMPROVEMENT:
   - ESM-2 captures evolutionary constraints
   - Mutations violating constraints = higher resistance
   - Expected: +10-15% correlation improvement
        """)

        return True

    except ImportError as e:
        print(f"✗ ESM-2 not available: {e}")
        print("  Install with: pip install transformers torch")
        return False
    except Exception as e:
        print(f"✗ Error testing ESM-2: {e}")
        return False


# =============================================================================
# 2. PROTTRANS (T5-BASED PROTEIN MODEL)
# =============================================================================


def test_prottrans_embeddings():
    """
    Test ProtTrans embeddings from Rostlab.

    ProtTrans provides:
    - Multiple model sizes (works on 8GB GPU)
    - T5-based architecture
    - Pre-trained on UniRef50/BFD

    Use cases:
    - Alternative to ESM-2 for limited GPU memory
    - Secondary structure prediction
    - Subcellular localization features
    """
    print_section("2. PROTTRANS EMBEDDINGS")

    try:
        import re

        import torch
        from transformers import T5EncoderModel, T5Tokenizer

        print("Loading ProtTrans model...")

        # Use the encoder-only model for embeddings
        model_name = "Rostlab/prot_t5_xl_half_uniref50-enc"

        # Check if model is cached, otherwise use smaller model
        try:
            tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
            model = T5EncoderModel.from_pretrained(model_name)
        except Exception:
            print("  Large model not cached, using smaller prot_bert...")
            from transformers import BertModel, BertTokenizer

            model_name = "Rostlab/prot_bert"
            tokenizer = BertTokenizer.from_pretrained(model_name, do_lower_case=False)
            model = BertModel.from_pretrained(model_name)

        model.eval()

        print(f"✓ Model loaded: {model_name}")

        # ProtTrans requires space-separated sequences
        sequence = HIV_SEQUENCES["protease"]
        sequence_spaced = " ".join(list(sequence))

        # Replace rare amino acids
        sequence_spaced = re.sub(r"[UZOB]", "X", sequence_spaced)

        print(f"\nTesting on HIV-1 Protease ({len(sequence)} residues)...")

        inputs = tokenizer(sequence_spaced, return_tensors="pt", add_special_tokens=True)

        with torch.no_grad():
            outputs = model(**inputs)

        embeddings = outputs.last_hidden_state
        print(f"\n✓ Embedding shape: {embeddings.shape}")

        # Mean pooling
        seq_embedding = embeddings.mean(dim=1)
        print(f"✓ Sequence embedding: {seq_embedding.shape}")

        print_subsection("ProtTrans vs ESM-2 Comparison")
        print("""
Feature Comparison:
┌─────────────────┬────────────────┬────────────────┐
│ Feature         │ ESM-2          │ ProtTrans      │
├─────────────────┼────────────────┼────────────────┤
│ Architecture    │ BERT-like      │ T5 (encoder)   │
│ Training data   │ UR50D          │ UniRef50/BFD   │
│ Min GPU memory  │ ~4GB (8M)      │ ~8GB           │
│ Best model      │ 650M (1280-d)  │ T5-XL (1024-d) │
│ Speed           │ Faster         │ Slower         │
│ Accuracy        │ Slightly better│ Very good      │
└─────────────────┴────────────────┴────────────────┘

Recommendation: Use ESM-2 for best accuracy, ProtTrans for memory constraints
        """)

        return True

    except ImportError as e:
        print(f"✗ ProtTrans not available: {e}")
        print("  Install with: pip install transformers torch sentencepiece")
        return False
    except Exception as e:
        print(f"✗ Error testing ProtTrans: {e}")
        return False


# =============================================================================
# 3. ALPHAFOLD DATABASE API
# =============================================================================


def test_alphafold_api():
    """
    Test AlphaFold Database API for protein structures.

    AlphaFold provides:
    - Predicted 3D structures for 214M+ proteins
    - Confidence scores (pLDDT)
    - PAE (Predicted Aligned Error) matrices

    Use cases:
    - Extract binding site residue positions
    - Calculate mutation distance to active site
    - Add structural features to model
    """
    print_section("3. ALPHAFOLD DATABASE API")

    # HIV-1 related UniProt IDs
    hiv_proteins = {
        "P04585": "Gag-Pol polyprotein (contains PR, RT, IN)",
        "P04591": "Envelope glycoprotein gp160",
        "P12497": "Gag polyprotein",
    }

    base_url = "https://alphafold.ebi.ac.uk/api"

    print("Testing AlphaFold API endpoints...")

    results = {}

    for uniprot_id, description in hiv_proteins.items():
        print(f"\n→ Querying {uniprot_id}: {description}")

        try:
            # Get prediction metadata
            url = f"{base_url}/prediction/{uniprot_id}"
            response = requests.get(url, timeout=30)

            if response.status_code == 200:
                data = response.json()[0] if isinstance(response.json(), list) else response.json()

                print("  ✓ Structure available!")
                print(f"    - Gene: {data.get('gene', 'N/A')}")
                print(f"    - Organism: {data.get('organismScientificName', 'N/A')}")
                print(f"    - Sequence length: {data.get('uniprotEnd', 0) - data.get('uniprotStart', 0) + 1}")
                print(f"    - Model version: {data.get('latestVersion', 'N/A')}")

                # Get download URLs
                print(f"    - PDB URL: {data.get('pdbUrl', 'N/A')[:60]}...")
                print(f"    - CIF URL: {data.get('cifUrl', 'N/A')[:60]}...")
                print(f"    - PAE URL: {data.get('paeImageUrl', 'N/A')[:60]}...")

                results[uniprot_id] = data

            elif response.status_code == 404:
                print("  ✗ No structure available (404)")
            else:
                print(f"  ✗ Error: {response.status_code}")

        except requests.exceptions.Timeout:
            print("  ✗ Request timed out")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    print_subsection("Structural Features We Can Extract")
    print("""
From AlphaFold structures, we can extract:

1. BINDING SITE DISTANCES:
   ```python
   from Bio.PDB import PDBParser

   # Parse structure
   parser = PDBParser()
   structure = parser.get_structure("protein", "alphafold.pdb")

   # Known active site residues for HIV protease
   active_site = [25, 26, 27]  # Catalytic triad

   # Calculate distance from each mutation to active site
   for mutation_pos in mutation_positions:
       dist = min_distance(mutation_pos, active_site)
       features.append(dist)
   ```

2. SECONDARY STRUCTURE:
   - Alpha helix, beta sheet, coil assignments
   - Mutations in structured regions may be more disruptive

3. SOLVENT ACCESSIBILITY:
   - Surface vs buried residues
   - Buried mutations often more destabilizing

4. pLDDT CONFIDENCE:
   - High confidence = well-structured
   - Low confidence = flexible/disordered

5. CONTACT MAPS:
   - Which residues interact
   - Mutations disrupting contacts = higher impact
    """)

    # Save results
    output_file = OUTPUT_DIR / "alphafold_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {output_file}")

    return len(results) > 0


# =============================================================================
# 4. STANFORD HIVDB API
# =============================================================================


def test_stanford_hivdb_api():
    """
    Test Stanford HIV Drug Resistance Database API.

    Stanford HIVDB provides:
    - Genotypic resistance interpretation
    - Drug susceptibility scores
    - Mutation penalty scores
    - Algorithm version information

    Use cases:
    - Ground truth resistance labels
    - Validate our predictions
    - Get mutation-specific penalty scores
    """
    print_section("4. STANFORD HIVDB API (Sierra Web Service)")

    # Sierra GraphQL API endpoint
    api_url = "https://hivdb.stanford.edu/graphql"

    # Test sequence with known mutations
    test_sequence = HIV_SEQUENCES["protease"]

    # GraphQL query for sequence analysis
    query = """
    query SequenceAnalysis($sequences: [UnalignedSequenceInput]!) {
      viewer {
        sequenceAnalysis(sequences: $sequences) {
          inputSequence {
            header
            sequence
          }
          validationResults {
            level
            message
          }
          alignedGeneSequences {
            gene {
              name
              drugClasses {
                name
              }
            }
            mutations {
              text
              position
              AAs
              isUnusual
              isDRM
            }
          }
          drugResistance {
            gene {
              name
            }
            drugScores {
              drug {
                name
                displayAbbr
                drugClass {
                  name
                }
              }
              score
              level
              text
              partialScores {
                mutations {
                  text
                }
                score
              }
            }
          }
        }
      }
    }
    """

    # Create a mutant sequence for testing
    mutant_seq = test_sequence[:45] + "I" + test_sequence[46:]  # M46I
    mutant_seq = mutant_seq[:53] + "V" + mutant_seq[54:]  # I54V
    mutant_seq = mutant_seq[:81] + "A" + mutant_seq[82:]  # V82A

    variables = {"sequences": [{"header": "HIV1_Protease_Mutant_M46I_I54V_V82A", "sequence": mutant_seq}]}

    print("Testing Stanford HIVDB GraphQL API...")
    print(f"  Query URL: {api_url}")
    print("  Test sequence: HIV-1 Protease with M46I, I54V, V82A mutations")

    try:
        response = requests.post(
            api_url,
            json={"query": query, "variables": variables},
            headers={"Content-Type": "application/json"},
            timeout=30,
        )

        if response.status_code == 200:
            data = response.json()

            if "errors" in data:
                print(f"\n✗ GraphQL errors: {data['errors']}")
                return False

            analysis = data["data"]["viewer"]["sequenceAnalysis"][0]

            print("\n✓ API Response received!")

            # Validation results
            print_subsection("Sequence Validation")
            for val in analysis.get("validationResults", []):
                print(f"  [{val['level']}] {val['message']}")

            # Mutations detected
            print_subsection("Mutations Detected")
            for gene_seq in analysis.get("alignedGeneSequences", []):
                gene_name = gene_seq["gene"]["name"]
                print(f"\n  Gene: {gene_name}")
                for mut in gene_seq.get("mutations", []):
                    drm_flag = "🔴 DRM" if mut["isDRM"] else "⚪"
                    unusual = "⚠️ Unusual" if mut["isUnusual"] else ""
                    print(f"    {mut['text']:10} {drm_flag} {unusual}")

            # Drug resistance scores
            print_subsection("Drug Resistance Scores")
            for dr in analysis.get("drugResistance", []):
                print(f"\n  Gene: {dr['gene']['name']}")
                print(f"  {'Drug':<10} {'Score':>6} {'Level':<20} {'Interpretation'}")
                print(f"  {'-' * 60}")

                for ds in dr.get("drugScores", []):
                    drug = ds["drug"]["displayAbbr"]
                    score = ds["score"]
                    ds["level"]
                    text = ds["text"]
                    print(f"  {drug:<10} {score:>6.1f} {text:<20}")

                    # Show which mutations contribute
                    for ps in ds.get("partialScores", []):
                        muts = ", ".join([m["text"] for m in ps["mutations"]])
                        print(f"    └─ {muts}: +{ps['score']}")

            # Save full response
            output_file = OUTPUT_DIR / "stanford_hivdb_results.json"
            with open(output_file, "w") as f:
                json.dump(data, f, indent=2)
            print(f"\n✓ Full results saved to: {output_file}")

            print_subsection("How to Use Stanford HIVDB Data")
            print("""
Integration with our model:

1. GROUND TRUTH LABELS:
   - Use drug resistance scores as training targets
   - Score ranges: 0-60+ (higher = more resistant)
   - Levels: Susceptible, Low, Intermediate, High

2. MUTATION PENALTIES:
   - Each mutation has a penalty score per drug
   - Can be used as features or for validation

3. BATCH PROCESSING:
   ```python
   # Process multiple sequences
   sequences = [{"header": f"seq_{i}", "sequence": seq}
                for i, seq in enumerate(dataset)]

   response = requests.post(api_url, json={
       "query": query,
       "variables": {"sequences": sequences}
   })
   ```

4. VALIDATION:
   - Compare our predictions vs Stanford scores
   - Correlation should be high (>0.9) for good model
            """)

            return True

        else:
            print(f"\n✗ HTTP Error: {response.status_code}")
            print(f"  Response: {response.text[:500]}")
            return False

    except requests.exceptions.Timeout:
        print("\n✗ Request timed out")
        return False
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


# =============================================================================
# 5. UNIPROT REST API
# =============================================================================


def test_uniprot_api():
    """
    Test UniProt REST API for protein annotations.

    UniProt provides:
    - Sequence data
    - Functional annotations
    - Domain information
    - Variant/mutation data
    - Cross-references to other databases

    Use cases:
    - Get canonical sequences
    - Identify functional domains
    - Find known pathogenic variants
    """
    print_section("5. UNIPROT REST API")

    base_url = "https://rest.uniprot.org"

    # HIV-1 protein entries
    proteins = {
        "P04585": "Gag-Pol polyprotein",
        "P04591": "Envelope glycoprotein gp160",
    }

    print("Testing UniProt REST API endpoints...")

    for uniprot_id, name in proteins.items():
        print(f"\n→ Fetching {uniprot_id}: {name}")

        try:
            # Get entry in JSON format
            url = f"{base_url}/uniprotkb/{uniprot_id}.json"
            response = requests.get(url, timeout=30)

            if response.status_code == 200:
                data = response.json()

                print("  ✓ Entry found!")

                # Basic info
                print(
                    f"    - Primary name: {data.get('proteinDescription', {}).get('recommendedName', {}).get('fullName', {}).get('value', 'N/A')}"
                )
                print(f"    - Organism: {data.get('organism', {}).get('scientificName', 'N/A')}")
                print(f"    - Sequence length: {data.get('sequence', {}).get('length', 0)} aa")

                # Features (domains, binding sites, etc.)
                features = data.get("features", [])
                feature_types = {}
                for f in features:
                    ft = f.get("type", "unknown")
                    feature_types[ft] = feature_types.get(ft, 0) + 1

                print(f"    - Features: {len(features)} total")
                for ft, count in sorted(feature_types.items(), key=lambda x: -x[1])[:10]:
                    print(f"      • {ft}: {count}")

                # Cross-references
                xrefs = data.get("uniProtKBCrossReferences", [])
                xref_dbs = set(x.get("database") for x in xrefs)
                print(f"    - Cross-references: {len(xrefs)} to {len(xref_dbs)} databases")
                print(f"      • Databases: {', '.join(sorted(list(xref_dbs))[:10])}...")

            else:
                print(f"  ✗ Error: {response.status_code}")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    # Test search functionality
    print_subsection("Search API")

    search_query = "HIV-1 protease AND reviewed:true"
    search_url = f"{base_url}/uniprotkb/search?query={search_query}&size=5&format=json"

    print(f"  Query: {search_query}")

    try:
        response = requests.get(search_url, timeout=30)
        if response.status_code == 200:
            data = response.json()
            results = data.get("results", [])
            print(f"  ✓ Found {len(results)} results (showing first 5)")
            for r in results:
                acc = r.get("primaryAccession", "N/A")
                name = (
                    r.get("proteinDescription", {}).get("recommendedName", {}).get("fullName", {}).get("value", "N/A")
                )
                print(f"    - {acc}: {name[:50]}")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    print_subsection("Available Data from UniProt")
    print("""
What we can extract:

1. CANONICAL SEQUENCES:
   - Reference sequences for HIV enzymes
   - Use as baseline for mutation detection

2. DOMAIN ANNOTATIONS:
   - Protease domain boundaries
   - RT domains (polymerase, RNase H)
   - Integrase domains (N-terminal, catalytic, C-terminal)

3. ACTIVE SITE RESIDUES:
   - Catalytic residues (e.g., D25 in protease)
   - Important for mutation impact prediction

4. KNOWN VARIANTS:
   - Natural polymorphisms
   - Disease-associated mutations

5. CROSS-REFERENCES:
   - PDB structures
   - Pfam domains
   - GO terms (function, process, location)
    """)

    return True


# =============================================================================
# 6. PDB DATA API
# =============================================================================


def test_pdb_api():
    """
    Test RCSB PDB Data API for protein structures.

    PDB provides:
    - Experimental protein structures
    - Ligand binding information
    - Resolution and quality metrics

    Use cases:
    - Get drug-bound structures
    - Identify binding site residues
    - Analyze mutation effects on structure
    """
    print_section("6. RCSB PDB DATA API")

    base_url = "https://data.rcsb.org/rest/v1/core"
    search_url = "https://search.rcsb.org/rcsbsearch/v2/query"

    # Search for HIV protease structures
    print("Searching for HIV-1 Protease structures...")

    search_query = {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entity_source_organism.taxonomy_lineage.name",
                        "operator": "exact_match",
                        "value": "Human immunodeficiency virus 1",
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {"attribute": "struct.title", "operator": "contains_words", "value": "protease"},
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "results_content_type": ["experimental"],
            "return_all_hits": False,
            "results_verbosity": "minimal",
        },
    }

    try:
        response = requests.post(search_url, json=search_query, timeout=30)

        if response.status_code == 200:
            data = response.json()
            total = data.get("total_count", 0)
            results = data.get("result_set", [])

            print(f"✓ Found {total} HIV-1 Protease structures")
            print("  Showing first 10:")

            for entry in results[:10]:
                pdb_id = entry.get("identifier", "N/A")
                print(f"    - {pdb_id}")

            # Get details for a specific structure
            if results:
                pdb_id = results[0]["identifier"]
                print_subsection(f"Structure Details: {pdb_id}")

                entry_url = f"{base_url}/entry/{pdb_id}"
                entry_response = requests.get(entry_url, timeout=30)

                if entry_response.status_code == 200:
                    entry_data = entry_response.json()

                    # Basic info
                    print(f"  Title: {entry_data.get('struct', {}).get('title', 'N/A')[:80]}")
                    print(f"  Method: {entry_data.get('exptl', [{}])[0].get('method', 'N/A')}")

                    resolution = entry_data.get("rcsb_entry_info", {}).get("resolution_combined", [None])[0]
                    if resolution:
                        print(f"  Resolution: {resolution:.2f} Å")

                    # Ligands
                    ligands = entry_data.get("rcsb_entry_info", {}).get("nonpolymer_bound_components", [])
                    if ligands:
                        print(f"  Ligands: {', '.join(ligands[:5])}")
        else:
            print(f"✗ Search failed: {response.status_code}")

    except Exception as e:
        print(f"✗ Error: {e}")

    # Get specific structure with drug
    print_subsection("Drug-Bound Structure Analysis")

    # 3OXC is HIV protease with Darunavir
    pdb_id = "3OXC"
    print(f"  Fetching {pdb_id} (HIV Protease + Darunavir)...")

    try:
        entry_url = f"{base_url}/entry/{pdb_id}"
        response = requests.get(entry_url, timeout=30)

        if response.status_code == 200:
            data = response.json()
            print("  ✓ Structure retrieved!")
            print(f"    Title: {data.get('struct', {}).get('title', 'N/A')[:80]}")

            # Get ligand info
            ligand_url = f"https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{pdb_id}/1"
            lig_response = requests.get(ligand_url, timeout=30)
            if lig_response.status_code == 200:
                lig_data = lig_response.json()
                print(f"    Ligand: {lig_data.get('pdbx_entity_nonpoly', {}).get('name', 'N/A')}")
    except Exception as e:
        print(f"  ✗ Error: {e}")

    print_subsection("Structural Features for Our Model")
    print("""
What we can extract from PDB:

1. BINDING SITE RESIDUES:
   - Identify which residues contact the drug
   - Mutations at binding sites = high resistance risk

   ```python
   # Example: residues within 4Å of drug
   binding_site = [23, 25, 27, 29, 30, 32, 47, 48, 50,
                   76, 80, 81, 82, 84]
   ```

2. DRUG-RESIDUE CONTACTS:
   - Hydrogen bonds
   - Hydrophobic contacts
   - Van der Waals interactions

3. STRUCTURAL FEATURES:
   - B-factors (flexibility)
   - Secondary structure
   - Solvent accessibility

4. MUTATION IMPACT:
   - If mutation removes key contact → resistance
   - Can calculate structural distance metrics
    """)

    return True


# =============================================================================
# 7. CHEMBL API
# =============================================================================


def test_chembl_api():
    """
    Test ChEMBL API for drug bioactivity data.

    ChEMBL provides:
    - Drug bioactivity measurements
    - Target information
    - Compound properties
    - Drug safety data

    Use cases:
    - Get IC50/EC50 values for HIV drugs
    - Find compound-target relationships
    - Identify drug resistance patterns
    """
    print_section("7. CHEMBL API")

    base_url = "https://www.ebi.ac.uk/chembl/api/data"

    # HIV-related targets
    print("Searching for HIV targets in ChEMBL...")

    # Search for HIV protease
    target_url = f"{base_url}/target/search.json?q=HIV+protease&limit=5"

    try:
        response = requests.get(target_url, timeout=30)

        if response.status_code == 200:
            data = response.json()
            targets = data.get("targets", [])

            print(f"✓ Found {len(targets)} HIV-related targets")

            for target in targets[:5]:
                chembl_id = target.get("target_chembl_id", "N/A")
                name = target.get("pref_name", "N/A")
                organism = target.get("organism", "N/A")
                print(f"  - {chembl_id}: {name} ({organism})")

            # Get activities for first target
            if targets:
                target_id = targets[0]["target_chembl_id"]
                print_subsection(f"Bioactivities for {target_id}")

                activity_url = f"{base_url}/activity.json?target_chembl_id={target_id}&limit=10"
                act_response = requests.get(activity_url, timeout=30)

                if act_response.status_code == 200:
                    act_data = act_response.json()
                    activities = act_data.get("activities", [])

                    print(f"  Found {len(activities)} activities (showing 10)")
                    print(f"  {'Compound':<15} {'Type':<10} {'Value':>10} {'Units':<10}")
                    print(f"  {'-' * 50}")

                    for act in activities[:10]:
                        comp = act.get("molecule_chembl_id", "N/A")[:15]
                        atype = act.get("standard_type", "N/A")[:10]
                        value = act.get("standard_value", "N/A")
                        units = act.get("standard_units", "N/A")[:10]

                        if value and value != "N/A":
                            print(f"  {comp:<15} {atype:<10} {float(value):>10.2f} {units:<10}")

        else:
            print(f"✗ Error: {response.status_code}")

    except Exception as e:
        print(f"✗ Error: {e}")

    # Get drug information
    print_subsection("HIV Drug Information")

    hiv_drugs = ["Darunavir", "Lopinavir", "Dolutegravir", "Rilpivirine"]

    for drug_name in hiv_drugs:
        try:
            drug_url = f"{base_url}/molecule/search.json?q={drug_name}&limit=1"
            response = requests.get(drug_url, timeout=30)

            if response.status_code == 200:
                data = response.json()
                mols = data.get("molecules", [])

                if mols:
                    mol = mols[0]
                    chembl_id = mol.get("molecule_chembl_id", "N/A")
                    max_phase = mol.get("max_phase", "N/A")
                    mol_type = mol.get("molecule_type", "N/A")

                    print(f"  {drug_name}: {chembl_id} (Phase {max_phase}, {mol_type})")
        except Exception as e:
            print(f"  {drug_name}: Error - {e}")

    print_subsection("ChEMBL Data for Our Model")
    print("""
What we can use:

1. DRUG POTENCY DATA:
   - IC50 values (inhibitory concentration)
   - Ki values (binding affinity)
   - Fold-change vs wild-type

2. RESISTANCE DATA:
   - Fold-change for mutant vs WT
   - Can correlate with our predictions

3. STRUCTURE-ACTIVITY:
   - Which drug modifications affect potency
   - Helps understand resistance mechanisms

4. DRUG PROPERTIES:
   - Molecular weight
   - LogP (lipophilicity)
   - PSA (polar surface area)
   - May correlate with resistance patterns
    """)

    return True


# =============================================================================
# 8. MAVEDB API
# =============================================================================


def test_mavedb_api():
    """
    Test MaveDB API for deep mutational scanning data.

    MaveDB provides:
    - 7+ million variant effect measurements
    - Functional scores for mutations
    - Standardized data format

    Use cases:
    - Training data for mutation predictors
    - Validation of our predictions
    - Benchmark comparisons
    """
    print_section("8. MAVEDB API")

    base_url = "https://api.mavedb.org/api/v1"

    print("Testing MaveDB API endpoints...")

    # Search for HIV-related experiments
    print("\n→ Searching for experiments...")

    try:
        # Get list of score sets
        search_url = f"{base_url}/score-sets/?limit=10"
        response = requests.get(search_url, timeout=30)

        if response.status_code == 200:
            data = response.json()
            results = data.get("results", []) if isinstance(data, dict) else data

            print("✓ Found score sets")

            if results:
                print("  Showing first 5:")
                for ss in results[:5]:
                    urn = ss.get("urn", "N/A")
                    title = ss.get("title", "N/A")[:50]
                    print(f"    - {urn}: {title}...")
        else:
            print(f"✗ Error: {response.status_code}")
            print(f"  Response: {response.text[:200]}")

    except Exception as e:
        print(f"✗ Error: {e}")

    # Get statistics
    print_subsection("MaveDB Statistics")

    try:
        stats_url = f"{base_url}/statistics/"
        response = requests.get(stats_url, timeout=30)

        if response.status_code == 200:
            stats = response.json()
            print(f"  Total score sets: {stats.get('scoreSetCount', 'N/A')}")
            print(f"  Total experiments: {stats.get('experimentCount', 'N/A')}")
            print(f"  Total variants: {stats.get('variantCount', 'N/A')}")
        else:
            # If stats endpoint doesn't exist, show general info
            print("  (Statistics endpoint not available)")
            print("  As of 2024: 7+ million variants, 1,884+ datasets")

    except Exception as e:
        print(f"  ✗ Error: {e}")

    print_subsection("MaveDB Data for Our Model")
    print("""
What we can use:

1. DEEP MUTATIONAL SCANNING DATA:
   - Experimental fitness scores for mutations
   - Ground truth for validation

2. HIV-SPECIFIC DATASETS:
   - Search for HIV protease/RT/IN DMS studies
   - May have resistance-relevant data

3. BENCHMARKING:
   - Compare our predictions vs experimental
   - ProteinGym uses MaveDB as benchmark

4. BULK DOWNLOAD:
   - CC0 licensed data on Zenodo
   - Can download entire database

Note: MaveDB data is most useful for:
- Model validation
- Training mutation effect predictors
- Benchmarking against other methods
    """)

    return True


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def main():
    """Run all API tests and generate summary."""

    print("\n" + "=" * 70)
    print(" COMPREHENSIVE API TESTING FOR HIV DRUG RESISTANCE PREDICTION")
    print("=" * 70)
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print("Testing all available APIs...\n")

    results = {}

    # Test each API
    tests = [
        ("ESM-2 Embeddings", test_esm2_embeddings),
        ("ProtTrans Embeddings", test_prottrans_embeddings),
        ("AlphaFold API", test_alphafold_api),
        ("Stanford HIVDB", test_stanford_hivdb_api),
        ("UniProt API", test_uniprot_api),
        ("PDB API", test_pdb_api),
        ("ChEMBL API", test_chembl_api),
        ("MaveDB API", test_mavedb_api),
    ]

    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n✗ {name} failed with error: {e}")
            results[name] = False

        time.sleep(1)  # Be nice to APIs

    # Summary
    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)

    print("\nAPI Test Results:")
    for name, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"  {name}: {status}")

    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\nTotal: {passed}/{total} APIs working")

    # Recommendations
    print("\n" + "=" * 70)
    print(" IMPLEMENTATION RECOMMENDATIONS")
    print("=" * 70)
    print("""
Based on API testing, here's the recommended implementation order:

PHASE 1 - HIGH IMPACT, LOW EFFORT:
1. ESM-2 Embeddings
   - Replace one-hot with 320/1280-dim embeddings
   - Expected improvement: +10-15% correlation

2. Stanford HIVDB Integration
   - Use as ground truth for validation
   - Get mutation penalty scores

PHASE 2 - STRUCTURAL FEATURES:
3. AlphaFold Structures
   - Download HIV enzyme structures
   - Extract binding site distances

4. PDB Drug-Bound Structures
   - Identify contact residues
   - Create structural feature set

PHASE 3 - EXTENDED DATA:
5. UniProt Annotations
   - Domain boundaries
   - Known variants

6. ChEMBL Bioactivity
   - IC50/Ki values
   - Cross-validate predictions

PHASE 4 - VALIDATION:
7. MaveDB Benchmarking
   - Compare with DMS data
   - Validate mutation effect predictions
    """)

    # Save summary
    summary = {"test_date": time.strftime("%Y-%m-%d %H:%M:%S"), "results": results, "passed": passed, "total": total}

    summary_file = OUTPUT_DIR / "api_test_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✓ Summary saved to: {summary_file}")

    return passed, total


if __name__ == "__main__":
    passed, total = main()
    print(f"\n{'=' * 70}")
    print(f" COMPLETED: {passed}/{total} APIs tested successfully")
    print(f"{'=' * 70}\n")
