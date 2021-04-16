import tkinter as tk
import tkinter.ttk as ttk
from pygubu.widgets.pathchooserinput import PathChooserInput
import paramiko
import h5py
import numpy as np
from os import listdir
from PIL import Image, ImageSequence
from matplotlib import pyplot as plt


class NewguiApp:
	def __init__(self, master=None):
		# build ui
		self.notebook3 = ttk.Notebook(master)
		self.frame3 = ttk.Frame(self.notebook3)
		self.pathChooserTrainImageStack = PathChooserInput(self.frame3)
		self.pathChooserTrainImageStack.configure(type='file')
		self.pathChooserTrainImageStack.grid(column='1', row='0')
		self.pathChooserTrainLabels = PathChooserInput(self.frame3)
		self.pathChooserTrainLabels.configure(type='file')
		self.pathChooserTrainLabels.grid(column='1', row='1')
		self.pathChooserTrainOutputDir = PathChooserInput(self.frame3)
		self.pathChooserTrainOutputDir.configure(type='file')
		self.pathChooserTrainOutputDir.grid(column='1', row='2')
		self.label1 = ttk.Label(self.frame3)
		self.label1.configure(text='Image Stack (.tif): ')
		self.label1.grid(column='0', row='0')
		self.label2 = ttk.Label(self.frame3)
		self.label2.configure(text='Labels (.h5)')
		self.label2.grid(column='0', row='1')
		self.label3 = ttk.Label(self.frame3)
		self.label3.configure(text='Output Dir: ')
		self.label3.grid(column='0', row='2')
		self.label4 = ttk.Label(self.frame3)
		self.label4.configure(text='# GPU: ')
		self.label4.grid(column='0', row='3')
		self.label5 = ttk.Label(self.frame3)
		self.label5.configure(text='# CPU: ')
		self.label5.grid(column='0', row='4')
		self.label6 = ttk.Label(self.frame3)
		self.label6.configure(text='Architecture: ')
		self.label6.grid(column='0', row='5')
		self.label7 = ttk.Label(self.frame3)
		self.label7.configure(text='Input Size: ')
		self.label7.grid(column='0', row='6')
		self.label8 = ttk.Label(self.frame3)
		self.label8.configure(text='Output Size: ')
		self.label8.grid(column='0', row='7')
		self.label9 = ttk.Label(self.frame3)
		self.label9.configure(text='In Planes: ')
		self.label9.grid(column='0', row='8')
		self.label10 = ttk.Label(self.frame3)
		self.label10.configure(text='Out Planes: ')
		self.label10.grid(column='0', row='9')
		self.label11 = ttk.Label(self.frame3)
		self.label11.configure(text='Loss Option: ')
		self.label11.grid(column='0', row='10')
		self.label12 = ttk.Label(self.frame3)
		self.label12.configure(text='Loss Weight: ')
		self.label12.grid(column='0', row='11')
		self.label13 = ttk.Label(self.frame3)
		self.label13.configure(text='Target Opt: ')
		self.label13.grid(column='0', row='12')
		self.label14 = ttk.Label(self.frame3)
		self.label14.configure(text='Weight Opt')
		self.label14.grid(column='0', row='13')
		self.label15 = ttk.Label(self.frame3)
		self.label15.configure(text='Pad Size: ')
		self.label15.grid(column='0', row='14')
		self.label16 = ttk.Label(self.frame3)
		self.label16.configure(text='LR_Scheduler: ')
		self.label16.grid(column='0', row='15')
		self.label17 = ttk.Label(self.frame3)
		self.label17.configure(text='Base LR: ')
		self.label17.grid(column='0', row='16')
		self.label18 = ttk.Label(self.frame3)
		self.label18.configure(text='Iteration Step: ')
		self.label18.grid(column='0', row='17')
		self.label19 = ttk.Label(self.frame3)
		self.label19.configure(text='Iteration Save: ')
		self.label19.grid(column='0', row='18')
		self.label20 = ttk.Label(self.frame3)
		self.label20.configure(text='Iteration Total: ')
		self.label20.grid(column='0', row='19')
		self.label21 = ttk.Label(self.frame3)
		self.label21.configure(text='Samples Per Batch: ')
		self.label21.grid(column='0', row='20')
		self.label22 = ttk.Label(self.frame3)
		self.label22.configure(text='Steps: ')
		self.label22.grid(column='0', row='21')
		self.numBoxTrainGPU = ttk.Spinbox(self.frame3)
		self.numBoxTrainGPU.configure(from_='0', increment='1', to='1000')
		_text_ = '''1'''
		self.numBoxTrainGPU.delete('0', 'end')
		self.numBoxTrainGPU.insert('0', _text_)
		self.numBoxTrainGPU.grid(column='1', row='3')
		self.numBoxTrainCPU = ttk.Spinbox(self.frame3)
		self.numBoxTrainCPU.configure(from_='1', increment='1', to='1000')
		_text_ = '''1'''
		self.numBoxTrainCPU.delete('0', 'end')
		self.numBoxTrainCPU.insert('0', _text_)
		self.numBoxTrainCPU.grid(column='1', row='4')
		self.entryTrainArchitecture = ttk.Entry(self.frame3)
		_text_ = '''unet_residual_3d'''
		self.entryTrainArchitecture.delete('0', 'end')
		self.entryTrainArchitecture.insert('0', _text_)
		self.entryTrainArchitecture.grid(column='1', row='5')
		self.entryTrainInputSize = ttk.Entry(self.frame3)
		_text_ = '''[112, 112, 112]'''
		self.entryTrainInputSize.delete('0', 'end')
		self.entryTrainInputSize.insert('0', _text_)
		self.entryTrainInputSize.grid(column='1', row='6')
		self.entryTrainOutputSize = ttk.Entry(self.frame3)
		_text_ = '''[112, 112, 112]'''
		self.entryTrainOutputSize.delete('0', 'end')
		self.entryTrainOutputSize.insert('0', _text_)
		self.entryTrainOutputSize.grid(column='1', row='7')
		self.numBoxTrainInPlanes = ttk.Spinbox(self.frame3)
		self.numBoxTrainInPlanes.configure(from_='1', increment='1', to='1000')
		_text_ = '''1'''
		self.numBoxTrainInPlanes.delete('0', 'end')
		self.numBoxTrainInPlanes.insert('0', _text_)
		self.numBoxTrainInPlanes.grid(column='1', row='8')
		self.numBoxTrainOutPlanes = ttk.Spinbox(self.frame3)
		self.numBoxTrainOutPlanes.configure(from_='1', increment='1', to='1000')
		_text_ = '''1'''
		self.numBoxTrainOutPlanes.delete('0', 'end')
		self.numBoxTrainOutPlanes.insert('0', _text_)
		self.numBoxTrainOutPlanes.grid(column='1', row='9')
		self.entryTrainLossOption = ttk.Entry(self.frame3)
		_text_ = '''[['WeightedBCE', 'DiceLoss']]'''
		self.entryTrainLossOption.delete('0', 'end')
		self.entryTrainLossOption.insert('0', _text_)
		self.entryTrainLossOption.grid(column='1', row='10')
		self.entryTrainLossWeight = ttk.Entry(self.frame3)
		_text_ = '''[[1.0, 1.0]]'''
		self.entryTrainLossWeight.delete('0', 'end')
		self.entryTrainLossWeight.insert('0', _text_)
		self.entryTrainLossWeight.grid(column='1', row='11')
		self.entryTrainTargetOpt = ttk.Entry(self.frame3)
		_text_ = '''['0']'''
		self.entryTrainTargetOpt.delete('0', 'end')
		self.entryTrainTargetOpt.insert('0', _text_)
		self.entryTrainTargetOpt.grid(column='1', row='12')
		self.entryTrainWeightOpt = ttk.Entry(self.frame3)
		_text_ = '''[['1', '0']]'''
		self.entryTrainWeightOpt.delete('0', 'end')
		self.entryTrainWeightOpt.insert('0', _text_)
		self.entryTrainWeightOpt.grid(column='1', row='13')
		self.entryTrainPadSize = ttk.Entry(self.frame3)
		_text_ = '''[56, 56, 56]'''
		self.entryTrainPadSize.delete('0', 'end')
		self.entryTrainPadSize.insert('0', _text_)
		self.entryTrainPadSize.grid(column='1', row='14')
		self.entryTrainLRScheduler = ttk.Entry(self.frame3)
		_text_ = '''"WarmupMultiStepLR"'''
		self.entryTrainLRScheduler.delete('0', 'end')
		self.entryTrainLRScheduler.insert('0', _text_)
		self.entryTrainLRScheduler.grid(column='1', row='15')
		self.numBoxTrainBaseLR = ttk.Spinbox(self.frame3)
		self.numBoxTrainBaseLR.configure(increment='.001', to='1000')
		_text_ = '''.01'''
		self.numBoxTrainBaseLR.delete('0', 'end')
		self.numBoxTrainBaseLR.insert('0', _text_)
		self.numBoxTrainBaseLR.grid(column='1', row='16')
		self.numBoxTrainIterationStep = ttk.Spinbox(self.frame3)
		self.numBoxTrainIterationStep.configure(from_='1', increment='1', to='10000000000')
		_text_ = '''1'''
		self.numBoxTrainIterationStep.delete('0', 'end')
		self.numBoxTrainIterationStep.insert('0', _text_)
		self.numBoxTrainIterationStep.grid(column='1', row='17')
		self.numBoxTrainIterationSave = ttk.Spinbox(self.frame3)
		self.numBoxTrainIterationSave.configure(from_='1', increment='1000', to='10000000000')
		_text_ = '''5000'''
		self.numBoxTrainIterationSave.delete('0', 'end')
		self.numBoxTrainIterationSave.insert('0', _text_)
		self.numBoxTrainIterationSave.grid(column='1', row='18')
		self.numBoxTrainIterationTotal = ttk.Spinbox(self.frame3)
		self.numBoxTrainIterationTotal.configure(from_='1', increment='5000', to='10000000000')
		_text_ = '''100000'''
		self.numBoxTrainIterationTotal.delete('0', 'end')
		self.numBoxTrainIterationTotal.insert('0', _text_)
		self.numBoxTrainIterationTotal.grid(column='1', row='19')
		self.numBoxTrainSamplesPerBatch = ttk.Spinbox(self.frame3)
		self.numBoxTrainSamplesPerBatch.configure(from_='1', increment='1', to='100000000000')
		_text_ = '''8'''
		self.numBoxTrainSamplesPerBatch.delete('0', 'end')
		self.numBoxTrainSamplesPerBatch.insert('0', _text_)
		self.numBoxTrainSamplesPerBatch.grid(column='1', row='20')
		self.entryTrainSteps = ttk.Entry(self.frame3)
		_text_ = '''(80000, 90000)'''
		self.entryTrainSteps.delete('0', 'end')
		self.entryTrainSteps.insert('0', _text_)
		self.entryTrainSteps.grid(column='1', row='21')
		self.buttonTrainTrain = ttk.Button(self.frame3)
		self.buttonTrainTrain.configure(text='Train')
		self.buttonTrainTrain.grid(column='0', row='24')
		self.buttonTrainTrain.configure(command=self.buttonTrainTrainPress)
		self.textTrainOutput = tk.Text(self.frame3)
		self.textTrainOutput.configure(height='10', width='50')
		_text_ = '''Model Output Will Be Shown Here'''
		self.textTrainOutput.insert('0.0', _text_)
		self.textTrainOutput.grid(column='0', columnspan='2', row='25')
		self.progressTrain = ttk.Progressbar(self.frame3)
		self.progressTrain.configure(orient='horizontal')
		self.progressTrain.grid(column='0', columnspan='2', row='23')
		self.buttonTrainRefresh = ttk.Button(self.frame3)
		self.buttonTrainRefresh.configure(text='Refresh Output')
		self.buttonTrainRefresh.grid(column='1', row='24')
		self.buttonTrainRefresh.configure(command=self.buttonTrainRefreshPress)
		self.label27 = ttk.Label(self.frame3)
		self.label27.configure(text='Username: ')
		self.label27.grid(column='0', row='27')
		self.label37 = ttk.Label(self.frame3)
		self.label37.configure(text='Password: ')
		self.label37.grid(column='0', row='28')
		self.entryTrainComputeClusterUsername = ttk.Entry(self.frame3)
		self.entryTrainComputeClusterUsername.grid(column='1', row='27')
		self.entryTrainComputeClusterPassword = ttk.Entry(self.frame3, show="*")
		self.entryTrainComputeClusterPassword.grid(column='1', row='28')
		self.label40 = ttk.Label(self.frame3)
		self.label40.configure(text='Below Information Will Run On Compute Cluster')
		self.label40.grid(column='0', columnspan='2', row='26')
		self.buttonTrainComputeClusterRun = ttk.Button(self.frame3)
		self.buttonTrainComputeClusterRun.configure(text='Run On Compute Cluster')
		self.buttonTrainComputeClusterRun.grid(column='0', columnspan='2', row='29')
		self.buttonTrainComputeClusterRun.configure(command=self.buttonTrainComputeClusterRunPress)
		self.frame3.configure(height='1000', width='1000')
		self.frame3.pack(side='top')
		self.notebook3.add(self.frame3, text='Train Network')
		self.frame4 = ttk.Frame(self.notebook3)
		self.pathChooserUseImageStack = PathChooserInput(self.frame4)
		self.pathChooserUseImageStack.configure(type='file')
		self.pathChooserUseImageStack.grid(column='1', row='0')
		self.pathChooserUseOutputDir = PathChooserInput(self.frame4)
		self.pathChooserUseOutputDir.configure(type='file')
		self.pathChooserUseOutputDir.grid(column='1', row='1')
		self.entryUseFileOutputName = ttk.Entry(self.frame4)
		_text_ = '''modelPred.h5'''
		self.entryUseFileOutputName.delete('0', 'end')
		self.entryUseFileOutputName.insert('0', _text_)
		self.entryUseFileOutputName.grid(column='1', row='2')
		self.entryUseInputSize = ttk.Entry(self.frame4)
		_text_ = '''[112, 112, 112]'''
		self.entryUseInputSize.delete('0', 'end')
		self.entryUseInputSize.insert('0', _text_)
		self.entryUseInputSize.grid(column='1', row='3')
		self.entryUseOutputSize = ttk.Entry(self.frame4)
		_text_ = '''[112, 112, 112]'''
		self.entryUseOutputSize.delete('0', 'end')
		self.entryUseOutputSize.insert('0', _text_)
		self.entryUseOutputSize.grid(column='1', row='5')
		self.entryUsePadSize = ttk.Entry(self.frame4)
		_text_ = '''[56, 56, 56]'''
		self.entryUsePadSize.delete('0', 'end')
		self.entryUsePadSize.insert('0', _text_)
		self.entryUsePadSize.grid(column='1', row='6')
		self.entryUseAugMode = ttk.Entry(self.frame4)
		_text_ = "'mean'"
		self.entryUseAugMode.delete('0', 'end')
		self.entryUseAugMode.insert('0', _text_)
		self.entryUseAugMode.grid(column='1', row='7')
		self.entryUseAugNum = ttk.Entry(self.frame4)
		_text_ = '''16'''
		self.entryUseAugNum.delete('0', 'end')
		self.entryUseAugNum.insert('0', _text_)
		self.entryUseAugNum.grid(column='01', row='8')
		self.numBoxUseSamplesPerBatch = ttk.Spinbox(self.frame4)
		self.numBoxUseSamplesPerBatch.configure(from_='1', increment='1', to='100000')
		_text_ = '''16'''
		self.numBoxUseSamplesPerBatch.delete('0', 'end')
		self.numBoxUseSamplesPerBatch.insert('0', _text_)
		self.numBoxUseSamplesPerBatch.grid(column='01', row='10')
		self.label23 = ttk.Label(self.frame4)
		self.label23.configure(text='Image Stack (.tif): ')
		self.label23.grid(column='0', row='0')
		self.label24 = ttk.Label(self.frame4)
		self.label24.configure(text='Output Dir: ')
		self.label24.grid(column='0', row='1')
		self.label25 = ttk.Label(self.frame4)
		self.label25.configure(text='File Output Name: ')
		self.label25.grid(column='0', row='2')
		self.label26 = ttk.Label(self.frame4)
		self.label26.configure(text='Input Size: ')
		self.label26.grid(column='0', row='3')
		self.label28 = ttk.Label(self.frame4)
		self.label28.configure(text='Output Size: ')
		self.label28.grid(column='0', row='5')
		self.label29 = ttk.Label(self.frame4)
		self.label29.configure(text='Pad Size')
		self.label29.grid(column='0', row='6')
		self.label30 = ttk.Label(self.frame4)
		self.label30.configure(text='Aug Mode: ')
		self.label30.grid(column='0', row='7')
		self.label31 = ttk.Label(self.frame4)
		self.label31.configure(text='Aug Num: ')
		self.label31.grid(column='0', row='8')
		self.label32 = ttk.Label(self.frame4)
		self.label32.configure(text='Samples Per Batch: ')
		self.label32.grid(column='0', row='10')
		self.label33 = ttk.Label(self.frame4)
		self.label33.configure(text='Stride: ')
		self.label33.grid(column='0', row='9')
		self.entry18 = ttk.Entry(self.frame4)
		_text_ = '''[56, 56, 56]'''
		self.entry18.delete('0', 'end')
		self.entry18.insert('0', _text_)
		self.entry18.grid(column='1', row='9')
		self.progressUse = ttk.Progressbar(self.frame4)
		self.progressUse.configure(orient='horizontal')
		self.progressUse.grid(column='0', columnspan='2', row='11')
		self.buttonUseRunModel = ttk.Button(self.frame4)
		self.buttonUseRunModel.configure(text='Run Model Prediction')
		self.buttonUseRunModel.grid(column='0', row='12')
		self.buttonUseRunModel.configure(command=self.buttonUseRunModelPress)
		self.buttonUseRefresh = ttk.Button(self.frame4)
		self.buttonUseRefresh.configure(text='Refresh Output')
		self.buttonUseRefresh.grid(column='1', row='12')
		self.buttonUseRefresh.configure(command=self.buttonUseRefreshPress)
		self.textUseOutput = tk.Text(self.frame4)
		self.textUseOutput.configure(height='10', width='50')
		_text_ = '''Output Will Go Here'''
		self.textUseOutput.insert('0.0', _text_)
		self.textUseOutput.grid(column='0', columnspan='2', row='13')
		self.frame4.configure(height='200', width='200')
		self.frame4.pack(side='top')
		self.notebook3.add(self.frame4, text='Use Network')
		self.frame5 = ttk.Frame(self.notebook3)
		self.label34 = ttk.Label(self.frame5)
		self.label34.configure(text='Model Output (.h5): ')
		self.label34.grid(column='0', row='0')
		self.label35 = ttk.Label(self.frame5)
		self.label35.configure(text='Ground Truth Label (.h5):')
		self.label35.grid(column='0', row='1')
		self.buttonEvaluateEvaluate = ttk.Button(self.frame5)
		self.buttonEvaluateEvaluate.configure(text='Evaluate')
		self.buttonEvaluateEvaluate.grid(column='0', columnspan='2', row='2')
		self.buttonEvaluateEvaluate.configure(command=self.buttonEvaluateEvaluatePress)
		self.pathChooserEvaluateLabels = PathChooserInput(self.frame5)
		self.pathChooserEvaluateLabels.configure(type='file')
		self.pathChooserEvaluateLabels.grid(column='1', row='1')
		self.pathChooserEvaluateModelOutput = PathChooserInput(self.frame5)
		self.pathChooserEvaluateModelOutput.configure(type='file')
		self.pathChooserEvaluateModelOutput.grid(column='1', row='0')
		self.frame5.configure(height='200', width='200')
		self.frame5.pack(side='top')
		self.notebook3.add(self.frame5, text='Evaluate Network')
		self.frame6 = ttk.Frame(self.notebook3)
		self.label36 = ttk.Label(self.frame6)
		self.label36.configure(text='Model Output (.h5): ')
		self.label36.grid(column='0', row='0')
		self.label38 = ttk.Label(self.frame6)
		self.label38.configure(text='Output Folder: ')
		self.label38.grid(column='0', row='1')
		self.pathChooserPredToolsModelOutput = PathChooserInput(self.frame6)
		self.pathChooserPredToolsModelOutput.configure(type='file')
		self.pathChooserPredToolsModelOutput.grid(column='1', row='0')
		self.pathChooserPredToolsOutputFolder = PathChooserInput(self.frame6)
		self.pathChooserPredToolsOutputFolder.configure(type='file')
		self.pathChooserPredToolsOutputFolder.grid(column='1', row='1')
		self.buttonOutputToolsPoints = ttk.Button(self.frame6)
		self.buttonOutputToolsPoints.configure(text='Make Point Clouds')
		self.buttonOutputToolsPoints.grid(column='0', row='2')
		self.buttonOutputToolsPoints.configure(command=self.buttonOutputToolsStatsPress)
		self.buttonOutputToolsMeshs = ttk.Button(self.frame6)
		self.buttonOutputToolsMeshs.configure(text='Make Meshs')
		self.buttonOutputToolsMeshs.grid(column='1', row='2')
		self.buttonOutputToolsMeshs.configure(command=self.buttonOutputToolsMeshsPress)
		self.buttonOutputToolsStats = ttk.Button(self.frame6)
		self.buttonOutputToolsStats.configure(text='Get Model Output Stats')
		self.buttonOutputToolsStats.grid(column='0', columnspan='2', row='3')
		self.buttonOutputToolsStats.configure(command=self.buttonOutputToolsStatsPress)
		self.textPredToolsOutput = tk.Text(self.frame6)
		self.textPredToolsOutput.configure(height='10', width='50')
		_text_ = '''Stats Will Be Output Here'''
		self.textPredToolsOutput.insert('0.0', _text_)
		self.textPredToolsOutput.grid(column='0', columnspan='2', row='4')
		self.frame6.configure(height='200', width='200')
		self.frame6.pack(side='top')
		self.notebook3.add(self.frame6, text='Model Output Tools')
		self.frame7 = ttk.Frame(self.notebook3)
		self.label39 = ttk.Label(self.frame7)
		self.label39.configure(text='Files To Visualize: ')
		self.label39.grid(column='0', row='0')
		self.pathChooserVisualizeFiles = PathChooserInput(self.frame7)
		self.pathChooserVisualizeFiles.configure(type='file')
		self.pathChooserVisualizeFiles.grid(column='1', row='0')
		self.buttonVisualizeVisualize = ttk.Button(self.frame7)
		self.buttonVisualizeVisualize.configure(text='Visualize')
		self.buttonVisualizeVisualize.grid(column='0', columnspan='2', row='2')
		self.buttonVisualizeVisualize.configure(command=self.buttonVisualizeVisualizePress)
		self.frame7.configure(height='200', width='200')
		self.frame7.pack(side='top')
		self.notebook3.add(self.frame7, text='Visualize')
		self.notebook3.pack(side='top')

		# Main widget
		self.mainwindow = self.notebook3

	def buttonTrainTrainPress(self):
		pass

	def buttonTrainRefreshPress(self):
		pass

	def sendSSHCommand(self, client, command, resultsText=""):
		stdin, stdout, stderr = client.exec_command(command)
		for line in stdout.readlines():
			resultsText += line
		return resultsText

	def buttonTrainComputeClusterRunPress(self):
		client = paramiko.SSHClient()
		client.load_system_host_keys()
		client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
		uname = self.entryTrainComputeClusterUsername.get()
		passw = self.entryTrainComputeClusterPassword.get()

		results = ""
		client.connect('hpcc.msu.edu', username=uname, password=passw)
		results = self.sendSSHCommand(client, 'qsub project/pytorch_connectomics/subDemo.sb', results)
		# results = self.sendSSHCommand(client, 'ls', results)
		# results = self.sendSSHCommand(client, 'qsub subDemo.sb', results)

		self.textTrainOutput.delete(1.0,"end")
		self.textTrainOutput.insert(1.0, results)
		

	def buttonUseRunModelPress(self):
		pass

	def buttonUseRefreshPress(self):
		pass

	def buttonEvaluateEvaluatePress(self):
		labelImage = self.pathChooserEvaluateLabels.entry.get()
		modelOutput = self.pathChooserEvaluateModelOutput.entry.get()
		labels = []
		im = Image.open(labelImage)
		for i, frame in enumerate(ImageSequence.Iterator(im)):
			framearr = np.asarray(frame)
			labels.append(framearr)
		labels = np.array(labels)

		h = h5py.File(modelOutput,'r')
		pred = np.array(h['vol0'][0])
		h.close()

		cutoffs = []
		ls = []
		ps = []
		percentDiffs = []
		precisions = []
		accuracies = []
		recalls = []
		for cutoff in range(0, 30, 1):
			cutoffs.append(cutoff)
			workingPred = np.copy(pred)
			workingPred[workingPred >= cutoff] = 255
			workingPred[workingPred != 255] = 0

			tp = np.sum((workingPred == labels) & (labels==255))
			tn = np.sum((workingPred == labels) & (labels==0))
			fp = np.sum((workingPred != labels) & (labels==0))
			fn = np.sum((workingPred != labels) & (labels==255))

			percentDiff = 1 - (np.count_nonzero(labels==255) - np.count_nonzero(workingPred==255))/np.count_nonzero(workingPred==255)
			percentDiffs.append(percentDiff)

			precisions.append(tp/(tp+fp))
			recalls.append(tp/(tp+fn))
			accuracies.append((tp + tn)/(tp + fp + tn + fn))

			ls.append(np.count_nonzero(labels))
			ps.append(np.count_nonzero(workingPred))
			del(workingPred)

		precisions = np.array(precisions)
		recalls = np.array(recalls)
		f1 = 2 * (precisions * recalls)/(precisions + recalls)

		plt.plot(cutoffs, precisions, label='precision')
		plt.plot(cutoffs, recalls, label='recall')
		plt.plot(cutoffs, f1, label='f1')
		#plt.plot(cutoffs, accuracies, label='accuracy')
		plt.plot(cutoffs, percentDiffs, label='percent differences')
		plt.legend()
		plt.grid()
		plt.show()



	def buttonOutputToolsStatsPress(self):
		pass

	def buttonOutputToolsMeshsPress(self):
		pass

	def buttonVisualizeVisualizePress(self):
		pass

	def run(self):
		self.mainwindow.mainloop()

if __name__ == '__main__':
	import tkinter as tk
	root = tk.Tk()
	app = NewguiApp(root)
	app.run()

