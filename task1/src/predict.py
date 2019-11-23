model.load_state_dict(torch.load('simpleNet_model/model.pkl.{}'.format(47)))
model.train(False)
run_epoch(1, False)
dataloader = DataLoader(dataset=testData,
                        batch_size=128,
                        shuffle=False,
                        collate_fn=testData.collate_fn,
                        num_workers=4)
trange = tqdm(enumerate(dataloader), total=len(dataloader), desc='Predict')
prediction = []
for i, (x, y) in trange:
    o_labels = model(x.to(device))
    o_labels = torch.argmax(o_labels, axis=1)
    prediction.append(o_labels.to('cpu'))

prediction = torch.cat(prediction).detach().numpy().astype(int)
