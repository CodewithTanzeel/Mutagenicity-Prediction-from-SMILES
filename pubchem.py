import requests

url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/aspirin/property/IsomericSMILES/TXT"

response = requests.get(url, timeout=10)

print(response.status_code)
print(response.text)