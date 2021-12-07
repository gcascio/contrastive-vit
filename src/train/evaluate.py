def evaluate(model, loader):
    total_samples = 0
    correct_classified = 0
    model.eval()

    for data, target in loader:
        output = model(data)
        prediction = output.max(1, keepdim=True)[1]
        correct_classified += prediction.eq(target.view_as(prediction)).sum()
        total_samples += data.size()[0]

    accuracy = 100.0 * correct_classified / total_samples

    return accuracy
