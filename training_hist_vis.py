import matplotlib.pyplot as plt
import pandas as pd

def training_hist_vis(hist, epochs):
	loss = hist.history['loss']
	val_loss = hist.history['val_loss']
	acc = hist.history['accuracy']
	val_acc = hist.history['val_accuracy']


	fig = plt.figure(figsize=(8,4))

	ax1 = fig.add_subplot(121)
	ax1.plot(loss,label='train_loss')
	ax1.plot(val_loss,label='val_loss')
	ax1.set_xlabel('Epoch')
	ax1.set_ylabel('Loss')
	ax1.set_title('Loss on Training and Validation Data')
	ax1.legend()

	ax2 = fig.add_subplot(122)
	ax2.plot(acc,label='train_accuracy')
	ax2.plot(val_acc,label='val_accuracy')
	ax2.set_xlabel('Epoch')
	ax2.set_ylabel('Accuracy')
	ax2.set_title('Accuracy on Training and Validation Data')
	ax2.legend()
	plt.tight_layout()
	plt.savefig('XXX.png', dpi=300)
	
	list1 = np.array([loss,acc,val_loss,val_acc])
	name1 = range(1,epochs)
	name2 = ['loss','accuracy','val_loss','val_accuracy']
	test = pd.DataFrame(columns=name1,index=name2,data=list1)
	test.to_csv('XXX.csv',encoding='gbk')

if __name__ == '__main__':
    print('None')
