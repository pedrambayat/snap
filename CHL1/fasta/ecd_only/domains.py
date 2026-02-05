import requests

def get_extracellular_sequence(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.json"
    response = requests.get(url).json()
    
    full_sequence = response['sequence']['value']
    
    # Filter features for "Topological domain" labeled "Extracellular"
    for feature in response.get('features', []):
        if feature['type'] == 'Topological domain' and 'Cytoplasmic' in feature['description']:
            start = feature['location']['start']['value']
            end = feature['location']['end']['value']
            # Python indexing starts at 0, UniProt at 1
            return full_sequence[start-1:end]
            
    return "Extracellular domain not found"

# Example: Get ECD for CHL1 (O00533) 
print(get_extracellular_sequence("O00533"))