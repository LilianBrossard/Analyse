import csv

def convert_csv(input_file, output_file):
    # Colonnes à garder (indices)
    columns_to_keep = [1,3,5,6,7,8,9,10,11,58,85,86,90,94]
    data = []

    # Lecture du CSV et stockage dans une liste de listes
    with open(input_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            data.append(row)

    # Filtrer les colonnes désirées
    filtered_data = []
    for row in data:
        filtered_row = [row[i] for i in columns_to_keep if i < len(row)]
        filtered_data.append(filtered_row)

    # Écriture du nouveau CSV
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(filtered_data)

# Exemple d'utilisation
if __name__ == "__main__":
    convert_csv('ROCKET_LEAGUE_WORLDS_21&22_DATA - ROCKET_LEAGUE_WORLDS_21&22_DATA.csv', 'output.csv')