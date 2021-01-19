def make_w():

    R=np.array(k)

    R_q=R.reshape(36,6,6)                 #R_q.shape => (36,6,6)

    pre_p=R.reshape(6,36,6)
    R_p=np.transpose(pre_p,(1,2,0))       #R_p.shape => (36,6,6)
    ###########################################################################
    R_p=torch.tensor(R_p)                 # numpy array 를 pytorch tensor 로 바꿔줌
    R_q=torch.tensor(R_q)           

    gap=nn.AvgPool2d(6)
    m_p=gap(R_p)
    m_q=gap(R_q)

    m_p=m_p.squeeze()
    m_p=m_p.type('torch.FloatTensor')

    m_q=m_q.squeeze()
    m_q=m_q.type('torch.FloatTensor')

    model = torch.nn.Sequential(
        nn.Linear(36, 6),
        nn.ReLU(),
        nn.Linear(6, 36)
    )

    w_p=model(m_p)
    w_q=model(m_q)
