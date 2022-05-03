
def save_csv(dataset, filepath):
    text = []
    text.append(','.join(list(dataset.columns.values)))
    for i in range(dataset.shape[0]):
        text.append(','.join(dataset.iloc[i]))
    with open(filepath, 'w') as f:
        f.write('\n'.join(text))
