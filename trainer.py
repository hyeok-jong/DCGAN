from locale import normalize
import time 
from tqdm import tqdm
import torch
import copy
import torch.optim as optim
import torch.nn as nn
from init_weight import initialize_weights


def get_lr(opt):         
    for param_group in opt.param_groups:
        return param_group['lr']
    
# There are no target (ground truth) data for GAN
# Rather one can set those as 0 and 1


# Get learning rate per every epoch. 
def get_lr(opt):         
    for param_group in opt.param_groups:
        return param_group['lr']
    
# There are no target (ground truth) data for GAN
# Rather one can set those as 0 and 1


def trainer(params):
    Generator         = params['Generator']
    Discriminator     = params['Discriminator']
    num_epochs        = params['num_epochs']
    loss_function     = params["loss_function"]
    train_dl          = params["train_dl"]
    lr                = params["learning_rate"]
    #lr_scheduler     = params["lr_scheduler"]
    # batch_size_init   = params["batch_size_init_only"]   # It changes!! for just last iteration
    result_dir        = params["result_dir"]
    device            = params["GPU"]

    # Model to GPU and initialize
    Generator = Generator.to(device)
    Generator.apply(initialize_weights)

    Discriminator = Discriminator.to(device)
    Discriminator.apply(initialize_weights)

    # Optimizer
    optimizer_D = optim.Adam(Discriminator.parameters(), lr = lr, betas = (0.5, 0.999))
    optimizer_G = optim.Adam(Generator.parameters(), lr = lr, betas = (0.5, 0.999))

    # Scheduler
    #lr_scheduler_D = ReduceLROnPlateau(optimizer_D, mode = "min", factor = 0.1, patience = 10)
    #lr_scheduler_G = ReduceLROnPlateau(optimizer_G, mode = "min", factor = 0.1, patience = 10)

    # Set dict for save losses
    # loss_history_batch = {'Generator_loss' : [], "Discriminator_loss" : []} *************This is for batch***************
    loss_history = {'Generator_loss' : [], "Discriminator_loss" : []}

    # Set dct for probabilitis
    # prob_history_batch = {"D_real" : [], "D_G_fake1" : [], "D_G_fake2" : []} *************This is for batch***************
    prob_history = {"D_real" : [], "D_G_fake1" : [], "D_G_fake2" : []}

    # It means number of batch training for whole dataset.
    iter_nums = len(train_dl)

    # Deepcopy is recomended. For more details one can see pytorch doc ; link below                                        
    # https://discuss.pytorch.org/t/copy-deepcopy-vs-clone/55022
    #best_generator_weight     = copy.deepcopy(Generator.state_dict())   
    #best_discriminator_weight = copy.deepcopy(Discriminator.state_dict())  

    # preset loss for save best model by loss
    #best_loss_G = float('inf')
    #best_loss_D = float('inf')

    start_time = time.time()



    # train() means "we gonna go train!" 
    # Why we need this?  for instance BN and Dropout there are difference between train and valid(test)
    # Actually, for GAN there is no valid during training, so this process is not necessary.
    Discriminator.train()  
    Generator.train()  

    # Fix noise to check generated image by same noise
    noise_fixed = torch.randn(16, 100, 1, 1, device = device)
    ''' 🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓
    noise_fixed = torch.randn(16, 100, 7, 7, device = device)
    🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓 '''

    # GAN doesn't need Valid
    # And Here, I coded loss to be saved every single epoch

    # Epoch, Batch, Iteration
    # There are len(train_dl.dataset) = 202599 images
    # Suppose that : Batch_size = 128
    # Then len(train_dl) = 1583 = [ { len(train_dl.dataset) // 128 } + 1]
    # Thus, for iterating whole dataset there are 1583 iterations.
    # For each iteration training 1 batch dataset is used.
    # And for 1-batch learning there are 128 images for parameter updating

    # Because iteration is initilized every epoch = line
    # Thus set total iteration as iteration_total
    iteration_total = 0

    # Fake images from fixed noise
    fake_image_list = []

    for epoch in range(num_epochs):
        # This is for 1-Epoch : It means model is trained by 1-whole dataset.

        # get current leraning rate and print it
        '''     *************This is for batch*************** 
        current_lr_D = get_lr(optimizer_D)
        current_lr_G = get_lr(optimizer_D)
        print(f'Epoch : {epoch}/{num_epochs-1}, current learning rate D / G = {current_lr_D, current_lr_G}')
        '''


        # Set losses ans probs to 0
        # This is mean value for 1-mini batch
        '''    *************This is for batch***************
        D_real_batch_sum = 0
        D_G_fake_1_batch_sum = 0
        D_G_fake_2_batch_sum = 0
        loss_Discriminator_batch_sum = 0
        loss_Generator_batch_sum = 0
        '''

        
        for iteration, images_batch in tqdm(enumerate(train_dl)):   
            # This is for 1-iteration : It means model trained by 1-mini_batch

            # For last iteration, Batch_size is different.
            batch_size = images_batch.shape[0]

            # Makes target data
            # One can think that it' oaky making target coded at first (means before for)
            # Warning!!!! For last batch training batch_size changed!!!!!!!!!
            
            
            
            real_target = torch.full(size = (batch_size,), fill_value = 1.0, dtype = torch.float, device = device )
            fake_target = torch.full(size = (batch_size,), fill_value = 0.0, dtype = torch.float, device = device )
            
            ''' 🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓
            real_target = torch.full(size = (batch_size*7*7,), fill_value = 1.0, dtype = torch.float, device = device )
            fake_target = torch.full(size = (batch_size*7*7,), fill_value = 0.0, dtype = torch.float, device = device )
            🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓 ''' 

            #---------     Train Discriminator    -------------------------------------------------------------------
            # zero grad /// opt.zero_grad() ???? ------>  https://discuss.pytorch.org/t/model-zero-grad-or-optimizer-zero-grad/28426
            Discriminator.zero_grad()



            # D( real_image )
            images_batch  = images_batch.to(device)
            real_batch = Discriminator(images_batch)         # [batch, 1, 1, 1]    Thus reshaping is needed for calculating loss
            D_real = real_batch.mean().item()                   # Means probabilty for Discriminator when real image came.

            # D( G(latent_vector)=fake_image )
            # Using .detach(), gradinets for Generator will be never calculated
            noise = torch.randn(batch_size, 100, 1, 1, device = device)
            
            ''' 🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓
            noise = torch.randn(batch_size, 100, 7, 7, device = device)
            🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓🤓 ''' 
            
            fake_image = Generator(noise)                    # Generated image
            fake_batch = Discriminator(fake_image.detach())    # [batch, 1, 1, 1]  
            D_G_fake_1 = fake_batch.mean().item()            # Means probabilty for Discriminator when fake image came.
            
            # For Discriminator loss see link below 
            # https://user-images.githubusercontent.com/78862026/162211986-4f27c990-c562-4aa7-b427-e456f8422d7f.png
            # Discriminator loss function is -[ log(D(real)) + log{1-D(G(fake))}]
            # This implies that, for real images D will be trained to output 1, and for fake images D to 0
            # Thus G will be trained to cheat D.
            # And for code it implemented by 3 steps.
            # Two steps will calculate gradient for D, and the other one for G.

            # This is loss for train D.
            # First for D, when real image came.
            loss_D_real = loss_function(real_batch.view(-1), real_target)
            loss_D_real.backward()   # Calculate gradient

            # Second for D, when fake image generated by Generator came into.
            loss_D_fake = loss_function(fake_batch.view(-1), fake_target)
            loss_D_fake.backward()

            # Loss sum ; It means loss for training discriminator
            # Of course calculate gradient at onec by loss_Discriminator.backward()
            loss_Discriminator = loss_D_real + loss_D_fake

            # Because it just involes Discriminator it updates only Discriminator
            optimizer_D.step()

            #---------     Train Generator    -------------------------------------------------------------------
            # Initialize Generator's gradient. 
            Generator.zero_grad()

            # Reuse fake_image line 111
            # But fake_batch in line 112 dosen't
            # Because one parameter update for Discriminator had procced
            # Therefore, It's not same.
            # And .detach() should not be used!!!!!!!!
            fake_batch = Discriminator(fake_image)
            D_G_fake_2 = fake_batch.mean().item() 

            # Gradient and update
            # Here G have to be trained to cheat D
            # So, label set to be real
            loss_Generator = loss_function(fake_batch.view(-1), real_target)
            loss_Generator.backward()

            # Parameter update
            optimizer_G.step()
            
            # Print losses ans probs
            if iteration_total % 100 == 0:
                print(f"\n[{epoch}/{num_epochs}][{iteration}/{iter_nums}] \nD Loss : {loss_Discriminator : .4f} \nG_Loss : {loss_Generator : .4f} \nD(real) : {D_real : .4f} \nD(G(fake))_1 : {D_G_fake_1 : .4f} \nD(G(fake))_2 : {D_G_fake_2 : .4f}")
                loss_history['Generator_loss'].append(loss_Generator.item())
                loss_history["Discriminator_loss"].append(loss_Discriminator.item())
                prob_history['D_real'].append(D_real)
                prob_history['D_G_fake1'].append(D_G_fake_1)
                prob_history['D_G_fake2'].append(D_G_fake_2)
                print(f"Time : {(time.time()-start_time)/60: .3f}min")
                print("-"*20, f"Total iterations 😀: {iteration_total}", "-"*20)

                # Save TXT

                with open(result_dir + "/Generator_loss.txt", "w", encoding = "UTF-8") as f:
                    for loss in loss_history['Generator_loss']:
                        f.write(str(loss) + "\n")
                with open(result_dir + "/Discriminator_loss.txt", "w", encoding = "UTF-8") as f:
                    for loss in loss_history['Discriminator_loss']:
                        f.write(str(loss) + "\n")
                with open(result_dir + "/D_real.txt", "w", encoding = "UTF-8") as f:
                    for prob in prob_history['D_real']:
                        f.write(str(prob) + "\n")
                with open(result_dir + "/D_G_fake1.txt", "w", encoding = "UTF-8") as f:
                    for prob in prob_history['D_G_fake1']:
                        f.write(str(prob) + "\n")
                with open(result_dir + "/D_G_fake2.txt", "w", encoding = "UTF-8") as f:
                    for prob in prob_history['D_G_fake2']:
                        f.write(str(prob) + "\n")


            # Save fake images form same vector
            if iteration_total % 300 == 0:
                with torch.no_grad():
                    fake_image_generated = Generator(noise_fixed).detach().cpu()
                fake_image_list.append(fake_image_generated)
            iteration_total += 1

            # Sum losses ans probs
            # This is summation 
            '''     *************This is for batch***************    
            loss_Discriminator_batch_sum += loss_Discriminator
            loss_Generator_batch_sum += loss_Generator
            D_real_batch_sum += D_real
            D_G_fake_1_batch_sum += D_G_fake_1
            D_G_fake_2_batch_sum += D_G_fake_2
            '''
            
            # Save Best model
            ###################################33


        # save losses and probs
        # Divide by iter nums -> calculate mean value
        # However it' not very accurate to do so.
        # For example whole dataset : 100 and batch size = 29
        # Then iteration will be 3 and left 13 datas.
        # So for last iteration batch size = 13  
        # That is there are 4 iterations and last one has different size.
        # Thus weighted sum is correct. But that error will be very slight.
        '''  *************This is for batch***************
        loss_history_batch['Generator_loss'].append(loss_Generator_batch_sum/iter_nums) 
        loss_history_batch["Discriminator_loss"].append(loss_Discriminator_batch_sum/iter_nums)
        prob_history_batch['D_real'].append(D_real_batch_sum/iter_nums)
        prob_history_batch['D_G_fake1'].append(D_G_fake_1_batch_sum/iter_nums)
        prob_history_batch['D_G_fake2'].append(D_G_fake_2_batch_sum/iter_nums)
        '''
    if epoch % 5 == 0:
        torch.save(Discriminator.state_dict(), result_dir + "/" + str(epoch) + "_epoch_Discriminator_params.pt")
        torch.save(optimizer_D.state_dict(), result_dir + "/" + str(epoch) + "_epoch_Discriminator_opt.pt")
        torch.save(Generator.state_dict(), result_dir + "/" + str(epoch) + "_epoch_Generator_params.pt")
        torch.save(optimizer_G.state_dict(), result_dir + "/" + str(epoch) + "_epoch_Generator_opt.pt")






    return Generator, loss_history, prob_history, fake_image_list


        





