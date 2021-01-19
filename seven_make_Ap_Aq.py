def make_Ap_Aq():
    w_p = w_p.type('torch.DoubleTensor')
    w_q = w_q.type('torch.DoubleTensor')

    bunmo_p=[0]
    A_p=[]

    for i in range (6):
        for j in range (6):
            p=torch.matmul(R_p[:,i,j],w_p)
            p=p.detach().numpy()
            temp=np.exp(p/0.025)
            bunmo_p=bunmo_p+temp                         # 하나씩 더해진 분모

    for i in range (6):
        for j in range (6):
            p=torch.matmul(R_p[:,i,j],w_p)
            p=p.detach().numpy()
            bunja=np.exp(p/0.025)
            A_p.append(bunja/bunmo_p)

    #print(A_p)
    #print(bunmo_p)

    ###################################################################################
    bunmo_q=[0]
    A_q=[]

    for i in range (6):
        for j in range (6):
            q=torch.matmul(R_q[:,i,j],w_q)
            q=q.detach().numpy()
            temp=np.exp(q/0.025)
            bunmo_q=bunmo_q+temp                         # 하나씩 더해진 분모

    for i in range (6):
        for j in range (6):
            q=torch.matmul(R_q[:,i,j],w_q)
            q=q.detach().numpy()
            bunja=np.exp(q/0.025)
            A_q.append(bunja/bunmo_q)

    #print(A_q)
    #print(bunmo_q)

    ###############################################################7 Ap Aq 구하기
