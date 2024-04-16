# import torch
# import numpy as np
#
# from EmbeddingNet import EmbeddingNet
# from JointNetDrift import Joint_Prediction
# import torch.optim as optim
#
# from TrainCentroidMatrixNet import FineTunedNet
# from preprocessing import LoadDriftData
# from config import args
# from torch.utils.data import TensorDataset,DataLoader
# from prototypical_batch_sampler import PrototypicalBatchSampler
# from StudentJointNet import Sub_Joint_Prediction
# from SmallJointNet import Small_Prediction
# from TeacherJointNet import Teacher_Joint_Prediction
# import json
# import os
# from torcheval.metrics.functional import multiclass_confusion_matrix
#
# np.random.seed(0)
# N_CLASSES = 4
# N_LAST_ELEMENTS = 5
# BATCH_SIZE = 40
# DTYPE_MAP = {
#     0: 'sudden',
#     1: 'gradual',
#     2: 'incremental',
#     3: 'normal'
# }
#
#
# def main():
#
#     ModelSelect = args.FAN # 'RNN', 'FAN', 'FNN','FQN','CNN'
#     # train_dataloader,test_dataloader=init_dataloader()
#     # splitted_test_set = split_drift_type(test_dataloader)
#     print('Checking if GPU is available')
#     use_gpu = torch.cuda.is_available()
#     use_gpu = False
#
#     # Set training iterations and display period
#     num_episode = args.num_episode
#
#     # Initializing prototypical net
#     print('Initializing Joint_Prediction net')
#     def train():
#         model = Joint_Prediction(use_gpu=use_gpu, Data_Vector_Length =args.Data_Vector_Length, ModelSelect=ModelSelect)
#         # optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.5)
#         optimizer = optim.Adam(model.parameters(), lr=args.lr)
#         lr_scheduler = init_lr_scheduler(optimizer)
#
#         train_loss = []
#         train_class_acc = []
#         train_loc_acc = []
#         test_loss = []
#         test_class_acc = []
#         test_loc_acc = []
#         centroid_matrix=torch.Tensor()
#         # Training loop
#         for i in range(num_episode):
#             model.train()
#             for batch_idx,data in enumerate(train_dataloader):
#                 optimizer.zero_grad()
#                 datax,datay,locy = data
#                 # locy = torch.unsqueeze(locy, 1)
#                 loss,class_acc,loc_acc,centroid_matrix = model(datax,datay,locy)
#                 # loss = loc_loss_fun(pre_loc_y,locy)
#                 loss.backward()
#                 optimizer.step()
#                 train_loss.append(loss.item())
#                 train_class_acc.append(class_acc.item())
#                 train_loc_acc.append(loc_acc.item())
#                 centroid_matrix=centroid_matrix
#
#             avg_loss = np.mean(train_loss)
#             avg_class_acc = np.mean(train_class_acc)
#             avg_loc_acc = np.mean(train_loc_acc)
#             print('{} episode,Avg Train Loss: {}, Avg Train Class Acc: {}, Avg Train loc Acc: {}'.format(
#                 i,avg_loss, avg_class_acc,avg_loc_acc))
#
#             lr_scheduler.step()
#
#         # Check whether the file exists
#         folder_path = "./input/Model/"+args.DATA_FILE
#         if not os.path.exists(folder_path):
#             os.makedirs(folder_path)
#         PATH = './input/Model/'+args.DATA_FILE+'/{name}_model_embeding.pkl'.format(name=ModelSelect)
#         torch.save(model.state_dict(), PATH)
#         # save centroid matrix
#         torch.save(centroid_matrix, './input/Model/' + args.DATA_FILE + '/{name}_centroid_matrix.pt'.format(name=ModelSelect))
#         # save loss
#         with open('./input/Model/' + args.DATA_FILE + '/{name}_loss.json'.format(name=ModelSelect), "w") as f:
#             json.dump({"train_loss":np.array(train_loss[len(train_loss)-5:]).sum()/len(train_loss[len(train_loss)-5:]),
#                        "train_class_acc":np.array(train_class_acc[len(train_class_acc)-5:]).sum()/len(train_class_acc[len(train_class_acc)-5:]),
#                        "train_loc_acc":np.array(train_loc_acc[len(train_loc_acc)-5:]).sum()/len(train_loc_acc[len(train_loc_acc)-5:])}, f)
#
#     # train()
#     # Test loop
#     centroid_matrix = torch.load(
#         './input/Model/' + args.DATA_FILE + '/{name}_centroid_matrix.pt'.format(name=ModelSelect))
#     model = Teacher_Joint_Prediction(use_gpu=use_gpu, Data_Vector_Length=args.Data_Vector_Length,
#                                              ModelSelect=ModelSelect)
#     PATH = './input/Model/' + args.DATA_FILE + '/' + ModelSelect + '_model_embeding.pkl'
#     model.load_state_dict(torch.load(PATH))
#
#     CM = torch.zeros(N_CLASSES,N_CLASSES)
#
#
#     test_class_acc = []
#     for batch_idx, data in enumerate(test_dataloader):
#         datax,datay,_ = data
#         y_pred, _, _ = model(datax, './input/Model/' + args.DATA_FILE)
#         y_pred = torch.t(y_pred)
#         # loss = loss_fn(y_pred, datay)
#         # test_loss.append(loss.item())
#         pred_labels = torch.argmax(y_pred, dim=1)
#         class_acc = (pred_labels == datay).sum().item() / len(datay)
#         test_class_acc.append(class_acc)
#         CM = torch.add(CM, multiclass_confusion_matrix(pred_labels, datay, num_classes=N_CLASSES))
#     avg_class_acc = np.mean(test_class_acc)
#     # avg_loc_acc = np.mean(test_loc_acc)
#     # save result
#     print(f'Confusion Matrix Pretrain:\n {CM}')
#     with open('./input/Model/' + args.DATA_FILE + '/{name}_result.json'.format(name=ModelSelect), "w") as f:
#         json.dump({"avg_class_acc":avg_class_acc,"avg_loc_acc":""}, f)
#
#     print(' Avg Test Class Acc: {}, Avg Test loc Acc: {}'.format(
#         avg_class_acc, ""))
#
#
# def knowledge_distillation():
#     '''
#     Knowledge Distillation
#     Distill a large model into a small one
#     '''
#     ModelSelect = args.FAN  # 'RNN', 'FAN','CNN', 'FNN','FQN','FAN'
#     train_dataloader, test_dataloader = init_dataloader()
#
#     print('Checking if GPU is available')
#     use_gpu = torch.cuda.is_available()
#     use_gpu = False
#
#     # Set training iterations and display period
#     num_episode = args.student_num_episode
#     #load teacher model
#     BASE_PATH = './input/Model/' + args.DATA_FILE
#     teather_model = Teacher_Joint_Prediction(use_gpu=use_gpu, Data_Vector_Length=args.Data_Vector_Length,
#                                 ModelSelect=ModelSelect)
#     PATH = BASE_PATH +'/'+ ModelSelect +'_model_embeding.pkl'
#     teather_model.load_state_dict(torch.load(PATH))
#     centroid_matrix = torch.load(
#         './input/Model/' + args.DATA_FILE + '/{name}_centroid_matrix.pt'.format(name=ModelSelect))
#
#     finetuning_model = FineTunedNet(centroid_matrix=centroid_matrix)
#     PATH = BASE_PATH + '/{name}_finetune_model_embeding.pkl'.format(name=ModelSelect)
#     finetuning_model.load_state_dict(torch.load(PATH))
#     # Initializing prototypical net
#     print('Initializing Joint_Prediction net')
#     sub_model = Sub_Joint_Prediction(Data_Vector_Length=args.Data_Vector_Length)
#     optimizer = optim.Adam(sub_model.parameters(), lr=args.student_lr)
#     lr_scheduler = init_lr_scheduler(optimizer)
#
#     train_loss = []
#     train_class_acc = []
#     train_loc_acc = []
#     test_loss = []
#     test_class_acc = []
#     test_loc_acc = []
#     # Training loop
#     for i in range(num_episode):
#         sub_model.train()
#         for batch_idx, data in enumerate(train_dataloader):
#             optimizer.zero_grad()
#             datax, datay, locy = data
#             #teacher model predict
#             # type_pred_T,loc_pred_T,loc_W = teather_model(datax, BASE_PATH)
#             embedded_x = teather_model.PrototypicalNet(datax)
#             type_pred_T = finetuning_model(embedded_x)
#             _, loc_pred_T, loc_W = teather_model(datax, BASE_PATH)
#             loss,type_loss,loc_loss,loc_acc,type_acc = sub_model(datax, datay,locy,type_pred_T,loc_pred_T,loc_W)
#             loss.backward()
#             optimizer.step()
#             train_loss.append(loss.item())
#             train_class_acc.append(type_acc)
#             train_loc_acc.append(loc_acc.item())
#
#         avg_loss = np.mean(train_loss)
#         avg_class_acc = np.mean(train_class_acc)
#         avg_loc_acc = np.mean(train_loc_acc)
#         print('{} episode,Avg Train Loss: {},Avg Train Class Acc: {}, Avg Train loc Acc: {}'.format(
#             i, avg_loss, avg_class_acc, avg_loc_acc))
#
#         lr_scheduler.step()
#
#     PATH = './input/Model/' + args.DATA_FILE + '/{name}_student_{T}_{M}_model_embeding.pkl'.format(name=ModelSelect,
#                                                                                                       T=args.distillation_T,
#                                                                                                       M=args.distillation_point_method)
#     torch.save(sub_model.state_dict(), PATH)
#     # save loss
#     with open('./input/Model/' + args.DATA_FILE + '/{name}_student_{T}_{M}_loss.json'.format(name=ModelSelect,
#                                                                                                       T=args.distillation_T,
#                                                                                                       M=args.distillation_point_method), "w") as f:
#         json.dump({"train_loss": train_loss, "train_class_acc": train_class_acc, "train_loc_acc": train_loc_acc}, f)
#     # Test loop
#     for batch_idx, data in enumerate(test_dataloader):
#         datax, datay, locy = data
#         embedded_x = teather_model.PrototypicalNet(datax)
#         type_pred_T = finetuning_model(embedded_x)
#         _, loc_pred_T,loc_W = teather_model(datax, BASE_PATH)
#         loss,type_loss,loc_loss,loc_acc,type_acc = sub_model(datax, datay,locy,type_pred_T,loc_pred_T,loc_W)
#         test_loss.append(loss.item())
#         test_class_acc.append(type_acc)
#         test_loc_acc.append(loc_acc.item())
#     avg_class_acc = np.mean(test_class_acc)
#     avg_loc_acc = np.mean(test_loc_acc)
#     # save result
#     with open('./input/Model/' + args.DATA_FILE + '/{name}_student_{T}_{M}_result.json'.format(name=ModelSelect,
#                                                                                                       T=args.distillation_T,
#                                                                                                       M=args.distillation_point_method), "w") as f:
#         json.dump({"avg_class_acc": avg_class_acc, "avg_loc_acc": avg_loc_acc}, f)
#
#     print(' Avg Test Class Acc: {}, Avg Test loc Acc: {}'.format(
#         avg_class_acc, avg_loc_acc))
#
# def small_model():
#     '''
#     small model
#     '''
#     ModelSelect = args.FAN  # 'RNN', 'FAN','CNN', 'FNN','FQN','FAN'
#     train_dataloader, test_dataloader = init_dataloader()
#
#     print('Checking if GPU is available')
#     use_gpu = torch.cuda.is_available()
#     # use_gpu = False
#
#     # Set training iterations and display period
#     num_episode = args.num_episode
#
#     # Initializing prototypical net
#     print('Initializing Joint_Prediction net')
#     sub_model = Small_Prediction(Data_Vector_Length=args.Data_Vector_Length)
#     optimizer = optim.Adam(sub_model.parameters(), lr=args.lr)
#     lr_scheduler = init_lr_scheduler(optimizer)
#
#     train_loss = []
#     train_class_acc = []
#     train_loc_acc = []
#     test_loss = []
#     test_class_acc = []
#     test_loc_acc = []
#     # Training loop
#     for i in range(num_episode):
#         sub_model.train()
#         for batch_idx, data in enumerate(train_dataloader):
#             optimizer.zero_grad()
#             datax, datay, locy = data
#             #teacher model predict
#             loss,type_loss,loc_loss,loc_acc,type_acc = sub_model(datax, datay,locy)
#             loss.backward()
#             optimizer.step()
#             train_loss.append(loss.item())
#             train_class_acc.append(type_acc)
#             train_loc_acc.append(loc_acc.item())
#
#         avg_loss = np.mean(train_loss)
#         avg_class_acc = np.mean(train_class_acc)
#         avg_loc_acc = np.mean(train_loc_acc)
#         print('{} episode,Avg Train Loss: {}, Avg Train Class Acc: {}, Avg Train loc Acc: {}'.format(
#             i, avg_loss, avg_class_acc, avg_loc_acc))
#
#         lr_scheduler.step()
#     PATH = './input/Model/' + args.DATA_FILE + '/{name}_small_model_embeding.pkl'.format(name=ModelSelect)
#     torch.save(sub_model.state_dict(), PATH)
#     # save loss
#     with open('./input/Model/' + args.DATA_FILE + '/{name}_small_loss.json'.format(name=ModelSelect), "w") as f:
#         json.dump({"train_loss": train_loss, "train_class_acc": train_class_acc, "train_loc_acc": train_loc_acc}, f)
#     # Test loop
#     for batch_idx, data in enumerate(test_dataloader):
#         datax, datay, locy = data
#         loss,type_loss,loc_loss,loc_acc,type_acc = sub_model(datax, datay,locy)
#         test_loss.append(loss.item())
#         test_class_acc.append(type_acc)
#         test_loc_acc.append(loc_acc.item())
#     avg_class_acc = np.mean(test_class_acc)
#     avg_loc_acc = np.mean(test_loc_acc)
#     # save result
#     with open('./input/Model/' + args.DATA_FILE + '/{name}_small_result.json'.format(name=ModelSelect), "w") as f:
#         json.dump({"avg_class_acc": avg_class_acc, "avg_loc_acc": avg_loc_acc}, f)
#
#     print(' Avg Test Class Acc: {}, Avg Test loc Acc: {}'.format(
#         avg_class_acc, avg_loc_acc))