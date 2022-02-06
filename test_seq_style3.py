import os, glob

#================== settings ==================
exp = 'QMUPD_model';epoch='200'
dataroot = 'examples'
gpu_id = '-1'

netga = 'resnet_style2_9blocks'
model0_res = 0
model1_res = 0
imgsize = 512
extraflag = ' --netga %s --model0_res %d --model1_res %d' % (netga, model0_res, model1_res)

#==================== command ==================
for vec in [[1,0,0],[0,1,0],[0,0,1]]:
    svec = '%d,%d,%d' % (vec[0],vec[1],vec[2])
    img1 = 'imagesstyle%d-%d-%d'%(vec[0],vec[1],vec[2])
    print('results/%s/test_%s/index%s.html'%(exp,epoch,img1[6:]))
    command = 'python test.py --dataroot %s --name %s --model test --output_nc 1 --no_dropout --model_suffix _A %s --num_test 1000 --epoch %s --style_control 1 --imagefolder %s --sinput svec --svec %s --crop_size %d --load_size %d --gpu_ids %s' % (dataroot,exp,extraflag,epoch,img1,svec,imgsize,imgsize,gpu_id)
    os.system(command)

