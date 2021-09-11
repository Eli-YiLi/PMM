import os
import numpy as np
from PIL import Image
import multiprocessing

categories = ['background','aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow',
              'diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
def do_python_eval(predict_folder, gt_folder, name_list, num_cls=21, input_type='png', threshold=1.0, printlog=False):
    TP = []
    P = []
    T = []
    for i in range(num_cls):
        TP.append(multiprocessing.Value('L', 0, lock=True))
        P.append(multiprocessing.Value('L', 0, lock=True))
        T.append(multiprocessing.Value('L', 0, lock=True))
    
    def compare(start,step,TP,P,T,input_type,threshold):
        for idx in range(start,len(name_list),step):
            name = name_list[idx]
            if input_type == 'png':
                predict_file = os.path.join(predict_folder,'%s.png'%name)
                predict = np.array(Image.open(predict_file)) #cv2.imread(predict_file)
                predict = predict[:,:]
            elif 'npy' in input_type:
                predict_file = os.path.join(predict_folder,'%s.npy'%name)
                predict_dict = np.load(predict_file, allow_pickle=True).item()
                h, w = list(predict_dict.values())[0].shape
                tensor = np.zeros((num_cls,h,w),np.float32)
                for key in predict_dict.keys():
                    v = predict_dict[key]
                    tensor[key+1] = predict_dict[key]
                tensor[0,:,:] = threshold 
                predict = np.argmax(tensor, axis=0).astype(np.uint8)

            gt_file = os.path.join(gt_folder,'%s.png'%name)
            gt = np.array(Image.open(gt_file))
            cal = gt<255
            mask = (predict==gt) * cal

            for i in range(num_cls):
                P[i].acquire()
                P[i].value += np.sum((predict==i)*cal)
                P[i].release()
                T[i].acquire()
                T[i].value += np.sum((gt==i)*cal)
                T[i].release()
                TP[i].acquire()
                TP[i].value += np.sum((gt==i)*mask)
                TP[i].release()
    p_list = []
    for i in range(8):
        p = multiprocessing.Process(target=compare, args=(i,8,TP,P,T,input_type,threshold))
        p.start()
        p_list.append(p)
    for p in p_list:
        p.join()
    IoU = []
    for i in range(num_cls):
        if T[i].value == 0:
            IoU.append(-1)
        else:
            IoU.append(TP[i].value/(T[i].value+P[i].value-TP[i].value))
    loglist = {}
    for i in range(num_cls):
        loglist[str(i)] = IoU[i] * 100
               
    #miou = np.mean(np.array(IoU))
    ious = np.array(IoU)
    miou = ious[ious>=0].mean()
    loglist['mIoU'] = miou * 100
    if printlog:
        for i in range(num_cls):
            if IoU[i] < 0:
                continue
            else:
                print('%11s:%7.3f%%'%(str(i),IoU[i]*100))
        print('\n======================================================')
        print('%11s:%7.3f%%'%('mIoU',miou*100))
    return loglist

def writedict(file, dictionary):
    s = ''
    for key in dictionary.keys():
        sub = '%s:%s  '%(key, dictionary[key])
        s += sub
    s += '\n'
    file.write(s)

def writelog(filepath, metric, comment):
    filepath = filepath
    logfile = open(filepath,'a')
    import time
    logfile.write(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))
    logfile.write('\t%s\n'%comment)
    writedict(logfile, metric)
    logfile.write('=====================================\n')
    logfile.close()


def evaluation(args):
    if args.type == 'npy':
        assert args.eval_thresh is not None or args.curve
    name_list = open(args.val_list).readlines()
    name_list = [n.strip().split()[0].split('/')[-1].split('.')[0] for n in name_list]
    if not args.curve:
        loglist = do_python_eval(args.predict_dir, args.gt_dir, name_list, args.num_cls, args.type, args.eval_thresh, printlog=True)
        writelog(args.eval_log_name, loglist, "")
    else:
        l = []
        for i in range(5,20):
            t = i/100.0
            loglist = do_python_eval(args.predict_dir, args.gt_dir, name_list, args.num_cls, args.type, t)
            l.append(loglist['mIoU'])
            print('%d/20 background score: %.3f\tmIoU: %.3f%%'%(i, t, loglist['mIoU']))
        writelog(args.eval_log_name, {'mIoU':l}, "")
