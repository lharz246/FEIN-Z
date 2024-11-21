


if __name__ == '__main__':
    # n = [[2,3,4],[5,6,7],[8,9,10]]
    # n = torch.FloatTensor(n)
    # print(n)
    # x = torch.diff(n,n=1,dim=0,prepend=torch.zeros(1,3))
    # print(x)
    # y = torch.diff(x,n=1,dim=0,prepend=torch.zeros(1,3))
    # print(y)
    # m = BCGG().cuda()
    # print(sys.path)
    # m2 = BCNetwork(1).cuda()
    # summary(m2, input_size=(128, 128), batch_size=1)
    # params = create_hparams()
    # # print(params.bc_embedding)
    # test_params = create_hparams1()
    # m0 = Encoder(params.audio_encoder).cuda()
    # m = AttentionHead(params.audio_attention).cuda()
    # m2 = BCNetwork(params).cuda()
    # print(m2)
    # loss = 100
    # print(os.getcwd())
    # t_dl, v_dl = prepare_dataloaders(params)
    # optim = torch.optim.Adam(m2.parameters(), lr=0.01, weight_decay=0.5)
    # for i, batch in enumerate(t_dl):
    #     previous_output = torch.zeros((64, 256, 300))
    #     m2.zero_grad()
    #     if not loss < 1.0:
    #         b = parse_batch(batch)
    #         audio, text, gesture, output_lengths, gate_padded = b
    #         # y = m0(audio, output_lengths)
    #         # y = m(y, y, y)
    #         y = m2((b, previous_output))
    #         print(f'output shape: {y.shape}')
    #         print(f'ground_truth: {gesture.shape}')
    #         loss = calculate_bc_loss({'joints': 1, 'velocity': 1, 'acceleration': 1}, y.transpose(1, 2), gesture)
    #         print(loss)
    #         loss.backward()
    #         optim.step()
    #         print(b)
    #     else:
    #         break

    for i in range(0,500,50):
        print(f' von {i} bis {i+100}')

    # rnd_in = torch.randn(256).cuda()
    # out = m2(rnd_in)
    # print(m2)
    # for o in out:
    #     print(o)
    #     print('\n')
    # print(m2)
    # m2 = BCNetwork(params).cuda()
    # hp = create_hparams()
    # summary(m2, input_size=(256, 256), batch_size=1)
    # summary(m2, (256, 256), 32)
    # print(m)
