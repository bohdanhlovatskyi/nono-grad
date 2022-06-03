
def test():
    x, y = np.array([1, 0, 0, 0, 1.]).reshape((1, 5)), np.array([1., 0.]).reshape((1, 2))
    X, Y = Tensor(x), Tensor(y)
    x, y = torch.Tensor(x), torch.Tensor(y)

    lll = np.random.rand(5, 4)
    L1 = Tensor(lll, requires_grad=True)
    l1 = torch.Tensor(copy.deepcopy(lll))

    lllll = np.random.rand(4, 2)
    L2 = Tensor(lllll, requires_grad=True)
    l2 = torch.Tensor(copy.deepcopy(lllll))
    
    x = x @ l1
    print("[first mul]: torch: ", x)
    X = Tensor(X.data @ L1.data)
    print("our: ", X.data)

    x = torch.relu(x)
    print("[first relu]: torch", x)
    X = X.relu()
    print("our: ", X.data)

    x = x @ l2
    print("[second mul]:  torch: ", x)
    X = Tensor(X.data @ L2.data)
    print("our: ", X.data)    

    ''' softmax is ok
    x = torch.softmax(x, dim=1)
    print("torch: ", x)
    X = X.softmax()
    print("out: ", X.data)
    '''

    x = torch.sigmoid(x)
    print("[sig]: torch: ", x)
    X = X.sigmoid()
    print("out: ", X.data)

    loss = torch.nn.MSELoss()
    print("[loss]: torch: ", loss(x, y))
    print("our: ", X.mse(Y))

    l, ol = loss(x, y), X.mse(Y)
    l.backward(torch.ones_like(l))
    ol.backward()

    print()

def test_mse():
    loss = torch.nn.MSELoss()
    a = torch.tensor([1.3643746, 2.37467237], requires_grad=True)
    b = torch.tensor([7.398497899, 4.43876743829])
    out = loss(a, b)
    out.backward()
    print(out)
    print(a.grad)

    an = np.array([1.3643746, 2.37467237]).reshape((2, 1))
    bn = np.array([7.398497899, 4.43876743829]).reshape((2, 1))
    a = Tensor(an, requires_grad=True)
    b = Tensor(bn)
    s = a.mse(b)
    print(s)
    s.backward()
    print(a.grad)

def test_matmul_and_mse():
    print("torch example")
    loss = torch.nn.MSELoss()
    a = torch.tensor([2., 3.], requires_grad=True)
    b = torch.tensor([6., 2.], requires_grad=True)

    z = a.matmul(b)
    true = torch.tensor([19.])
    out = loss(z, true)
    out.backward()
    print(out)
    print(a.grad)
    print(b.grad)

    print("our example")
    print()

    an = np.array([2., 3.]).reshape((2, 1))
    bn = np.array([6., 2.]).reshape((2, 1))
    a = Tensor(an.T, requires_grad=True)
    b = Tensor(bn, requires_grad=True)

    z = a @ b
    true = Tensor(np.array([19.]))
    out = z.mse(true)
    out.backward()
    print(out)
    print(a.grad)
    print(b.grad)

def test_xor():
    print("torch example")
    loss = torch.nn.MSELoss()
    first_layer = np.matrix('2. 3.; 4. 5')
    second_layer = np.matrix('6.; 7.')
    x = np.matrix('0., 1.')
    gt = np.matrix('1.')
    input = torch.tensor(x)
    first_layer_torch = torch.tensor(first_layer, requires_grad=True)
    second_layer_torch = torch.tensor(second_layer, requires_grad=True)

    w = input.matmul(first_layer_torch)
    w = w.matmul(second_layer_torch)
    true = torch.tensor(gt)
    out = loss(w, true)
    out.backward()
    print(out)
    print(first_layer_torch.grad)
    print(second_layer_torch.grad)

    print("our example")
    print()

    input_our = Tensor(x)
    first_layer_our = Tensor(first_layer, requires_grad=True)
    second_layer_our = Tensor(second_layer, requires_grad=True) 
    w = input_our @ first_layer_our
    w = w @ second_layer_our
    true = Tensor(gt)
    out = w.mse(true)
    out.backward()
    print(out)
    print(first_layer_our.grad)
    print(second_layer_our.grad)
