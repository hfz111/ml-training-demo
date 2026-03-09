from dataset import RandomDataset

def test_dataset_output():

    dataset = RandomDataset()

    x, y = dataset[0]

    assert x.shape[0] == 10
