# SFPSeg
环境搭建过程及遇到的问题：
步骤：
1-使用conda create -n SFPSeg python=3.10创建虚拟环境；
2-激活虚拟环境：
conda activate SFPSeg
3-安装pytorch:
 pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 torchaudio==2.0.2+cu118 -f https://download.pytorch.org/whl/torch_stable.html
4-注释掉requirements.txt中的关于pytorch的安装
5-使用 pip install -r requirements.txt安装要求的依赖库
6-构建selective_scan以及后续遇到的问题及解决办法见如下：

Q1:
ModuleNotFoundError: No module named 'DCNv3'

这是因为需要根据readme中的：
cd models/ops_dcnv3
   sh ./make.sh
编辑DCNv3，但编译过程会遇到以下问题：

Q2:
RuntimeError:
The detected CUDA version (12.2) mismatches the version that was used to compile
PyTorch (11.8). Please make sure to use the same CUDA versions.

Re:根据报错的位置定位到对应的函数，是在cpp_extension.py文件中的
    def build_extensions(self) -> None:
        compiler_name, compiler_version = self._check_abi()

        cuda_ext = False
        extension_iter = iter(self.extensions)
        extension = next(extension_iter, None)
        while not cuda_ext and extension:
            for source in extension.sources:
                _, ext = os.path.splitext(source)
                if ext == '.cu':
                    cuda_ext = True
                    break
            extension = next(extension_iter, None)

        # if cuda_ext and not IS_HIP_EXTENSION:
        #     _check_cuda_version(compiler_name, compiler_version)

将  if cuda_ext and not IS_HIP_EXTENSION:
             _check_cuda_version(compiler_name, compiler_version)函数注释掉，这是因为编译过程中有函数在检查版本是否一致，然后注释掉后，可以顺利使用兼容的Pytorch进入编译的下一阶段。

Q3:
import selective_scan_cuda_oflex_rh
ModuleNotFoundError: No module named 'selective_scan_cuda_oflex_rh'

RE3:
这是因为之前直接从github上下载对应pytorch和python版本的mamba，然后安装的，这导致本项目中使用的块没有，需要根据readme中的：
cd models/selective_scan && pip install .
进行构建，构建完后在目录下会自动生成这样的文件：
selective_scan_cuda_oflex_rh.cpython-310-x86_64-linux-gnu.so
但是在构建过程中会出现问题4：Q4

Q4:

File "<string>", line 17, in <module>
      ModuleNotFoundError: No module named 'torch'
      [end of output]

  note: This error originates from a subprocess, and is likely not a problem with pip.
ERROR: Failed to build 'file:///../models/selective_scan' when getting requirements to build wheel

RE4:实际上已经安装了pytorch，但是报以上错误
解决办法：cd models/selective_scan
Then use:
pip install -v -e . --no-build-isolation

Q5:
File "../models/ops_dcnv3/functions/dcnv3_func.py", line 53, in forward
    output = DCNv3.dcnv3_forward(*args)
RuntimeError: Not implemented on the CPU

RE：这是因为DCNv3没有在CPU上实现，因此需要修改代码，将模型与数据都放入到GPU。
if __name__=="__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DAMamba_T()
    x=torch.randn([2,3,512,512])
    encoder = Encoder(model)
    encoder.to(device)
    x = x.to(device)
    outlist = encoder(x)
    for out in outlist:
        print(out.shape)

根据以上步骤及方法实现了DAMamba模型的环境搭建及测试。
torch.Size([2, 80, 128, 128])  #stage1 output
torch.Size([2, 160, 64, 64])   #stage2 output
torch.Size([2, 320, 32, 32])   #stage3 output
torch.Size([2, 512, 16, 16])   #stage4 output

Q6:Run Error
return F.conv2d(x, self.conv.weight + self.conv_constant, self.conv.bias, stride=1,
RuntimeError: Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! (when checking argument for argument weight in method wrapper_CUDA___conv_depthwise2d)
RE: this because we use list in code but do not move it to the same device after operation
specifically, in builder.py line 87 and 88,
original code is :
decode_out = self.decoder[i](outs[i])
segout = self.seghead[i](decode_out)
updated is:
decode_out = self.decoder[i].to(rgb.device)(outs[i])
segout = self.seghead[i].to(rgb.device)(decode_out)

Q7:eturn torch.binary_cross_entropy_with_logits(input, target, weight, pos_weight, reduction_enum)
RuntimeError: result type Float can't be cast to the desired output type Long
RE: this is because the data type of target is torch.float32 or torch.float64, the original code is:
loss = self.criterion(out1, label.long())
loss += self.criterion(out2, label.long())
loss += self.criterion(out3, label.long())
loss += self.criterion(out4, label.long())
update the codes,as follows:
loss = self.criterion(out1, label.float())
loss += self.criterion(out2, label.float())
loss += self.criterion(out3, label.float())
loss += self.criterion(out4, label.float())

Q8: 当新复制一个项目到一个路径后，可能出现一下问题，
import selective_scan_cuda_oflex_rh
ModuleNotFoundError: No module named 'selective_scan_cuda_oflex_rh'
解决办法：cd models/selective_scan
Then use:
pip install -v -e . --no-build-isolation
跟Q4的解决办法一样，即可解决

Train:
python train.py
Test:
python test_pred_out.py


