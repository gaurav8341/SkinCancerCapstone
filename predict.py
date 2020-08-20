def predict(path, modelpath = "D:/Github/SkinCancerCapstone/models/resnet101.pth"):
    model = torch.load(modelpath)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    img = Image.open(path)
    img = test_transform(img).float()
    img = Variable(img, requires_grad=False)
    img = img.unsqueeze(0).to(device)
    output = model(img)
    print(output)
    m = nn.Softmax(dim = 1)
    op = m(output)
    op = op.cpu().detach().numpy()[0]
    print(op)
    opind = op.argsort()[-3:][::-1]
    print(opind)
#     prediction = output.max(1, keepdim=True)[1].tolist()
    li =[]
    for p in opind:
        li.append((lesion_type_dict[p], op[p]))
    return li