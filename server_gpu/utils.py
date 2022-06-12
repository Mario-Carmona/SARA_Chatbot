
def save_csv(dataset, filepath):
    text = []
    text.append(','.join(list(dataset.columns.values)))
    for i in range(dataset.shape[0]):
        line = []
        for elem in dataset.iloc[i]:
            #new_elem = elem.replace("\"", "'")
            line.append(f"\"{elem}\"")
        text.append(','.join(line))
    with open(filepath, 'w') as f:
        f.write('\n'.join(text))
