
def save_csv(dataset, filepath):
    text = []
    text.append(','.join(list(dataset.columns.values)))
    for i in range(dataset.shape[0]):
        line = [f"\"{elem}\"" for elem in dataset.iloc[i]]
        text.append(','.join(line))
    with open(filepath, 'w') as f:
        f.write('\n'.join(text))
