from Bio import Entrez


def get_gene_name_by_entrez_id(entrez_id):
    Entrez.email = "your_email@example.com"  # Always tell NCBI who you are
    handle = Entrez.efetch(db="gene", id=entrez_id, retmode="xml")
    records = Entrez.read(handle)

    # The gene name can usually be found in the "Gene-ref" field, under "locus"
    try:
        gene_name = records[0]["Entrezgene_gene"]["Gene-ref"]["Gene-ref_locus"]
        return gene_name
    except (KeyError, IndexError):
        return None


entrez_id = 5296
gene_name = get_gene_name_by_entrez_id(entrez_id)

if gene_name:
    print(f"The gene name for Entrez ID {entrez_id} is {gene_name}.")
else:
    print(f"No gene name found for Entrez ID {entrez_id}.")

