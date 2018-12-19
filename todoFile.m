%% all fail

%% partial pass

%% all pass
runtests('NNTest') % just warnings from precision mismatches
runtests('instNormLayerTest') 
runtests('convCuDNN2DTest') % - no mexcuda running

runtests('MegaNetTest');
runtests('singleLayerTest') % testGetJYOp tempermental
runtests('normLayerTest')
runtests('batchNormLayerTest')
runtests('tvNormLayerTest') 
runtests('doubleSymLayerTest')
runtests('doubleLayerTest')
runtests('affineScalingLayerTest') % CUDA issues
runtests('ResNNTest');
runtests('LeapFrogNNTest');
runtests('convMCNTest');
runtests('connectorTest');
runtests('DoubleHamiltonianNNTest');
runtests('IntegratorTest');
runtests('ConvFFTTest');
runtests('denseTest'); % testAdjoint tempermental
runtests('kernelTest');
runtests('scalingKernelTest');
runtests('sparseKernelTest');
runtests('convFFTTest') % MinExample
tb = runtests('layerTest');

%%
ECNN_MNIST_tf
EResNN_Peaks
EResNN_Circle


ECNN_CIFAR10_tf
E_ResNN_MNIST