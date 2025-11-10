# Dual-side Sparse Tensor Core

# Summary

| **Previous work** | this work’s approach | challenge | solution | scheme |
| --- | --- | --- | --- | --- |
| Current Tensor Core supports only weight sparsity | Supports activation sparsity on Tensor Core.
(first in GPU) | activation sparsity is un-predictable | BITMAP-Based SpGEMM | - Outer-product-based operation (dense multiplication and gather/scatter-based accumulation)
- two-level bitmap design for small local buffer |
| Several ASIC works support dual-side sparsity. But, SpGEMM only. | Provides SpCONV | im2col overhead. Also, im2col for sparse activation introduces random access. | Dual-Side Sparse Convolution | - outer-product friendly im2col
- bitmap-based sparse im2col |
| 1. different between SpGEMM vs SpCONV : explict im2col 

2. cost of im2col : affected by access to Shared MEM. (we don’t want to use register file) |  | 1. activation sparsity : online. prefer to do it online.

cf) NVIDIA 4:2 pruning. off-line |  |  |

# Background

## SpCONV

- DNN acceleration library

cuDNN applys im2col for CONV on Tensor cores. cuDNN uses the implicit im2col not to expand the memory footprint. (implicit im2col : original feature map is on global memory and gather them to on-chip memory using address calculation)

The im2col is mainly for activation (GUESS : weight can be processed off-line). So far, when utilizing the weight sparsity, the im2col is based on dense format. 

# Proposed Work

## BITMAP-based SpGEMM

- Dis-advantages of inner-product-based Tensor Core on dual-side sparsity

![image.png](image.png)

Under the current Tensor Core design, B(activation matrix)’s sparsity produces an under-utilization problem in Tensor Core (’Not used’ in Fig 3(c)). This damages the parallelism of dot products. 

Also, prior ASIC papers proposed several methods. But, the cost of those peripherals are considerable overhead to Tensor Cores. (propotional to large TC die-size)

- Advantages of outer-product-based Tensor Core

![image.png](image%201.png)

If the sparse vectors are condensed, then vector-vector outer product would be condensed (full util, Fig 4(c) ). Also, some of the unnecessary computations would be skipped. 

![image.png](image%202.png)

The gain of this process is not accumulation, but having less multiplication. 

More concretly, Av and Bv are tilied in 8x16x1. For the 1 colum of Av and for the 1 row of Bv, one 8x16x1 constitutes a step (Fig 5). So, as depicked in Fig 5, there are 8 steps when the col of Av is 32 and row of Bv is 32. Among 8 steps, 5 steps are skipped. 

After several steps of 8x16x1, the accumulation happens over K steps. 

(By the way, the 128 (=8x16x1) is from this; a warp controls two tensor cores simultaneously. A tensor core can complete 4x4x4 dense matrix multiplication per cycle which means that there are 64 multipliers and 64 accumulators. So, for a warp, we can assume that 128 multipliers and 128 accumulators are equipped.)

- SpGEMM on device (chellenge)

The result(partial output) of outer product is big. So, addressing the valid element(randomly distributed non-zero element) in the big partial output matrix woud exceed the size of local buffer. So, the author proposed two level bit-map (warp-bitmap, element-bitmap).

![image.png](image%203.png)

Matrix A, B = condensed globally → so that column & row bitmap are globally large.

Column Values, Row Value = condensed 

Column bitmap, Row bitmap = points to large partial matrix

![image.png](image%204.png)

Two-level bitmap = warp-bitmap, element-bitmap

Matrix A, B = condensed locally (warp-level; 32x32x16) 

Column Values, Row Value = condensed locally (within a warp-level)

Column bitmap, Row bitmap = points to smaller partial matrix with warp

So, in this setting, one thread block needs three tuples (warp-bitmap, element-bitmap, and value).

## Dual-Side Sparse Convolution

To leverage their own proposed SpGEMM, the authors invented a novel sparse im2col which is better for outer-product (better for re-use, implicit re-arranging in register). 

![image.png](image%205.png)

im2col for inner product  = normal im2col

im2col for outer product = (black dotted line) sliding window orderred im2col (data re-use)

![image.png](image%206.png)

for (a) left

bit-map at global memory

bit-map encoded input feature map at global memory

The bitmap-encoded input feature maps are stored in GMEM (RAM; GDDR or HBM). And then, load the bitmap and values into registers to do ‘apply mask’, ‘shift left’, and ‘pop count’. 

Those ‘apply mask’, ‘shift left’, and ‘pop count’ operations might be done via integer cuda cores using instructions like LD, AND, OR, SHL, etc. 

cf. The **matrix multiply–accumulate** itself runs on the **Tensor Cores**.

## Outer-product Tensor Core

![image.png](image%207.png)

- a Sub Core
    - math function units
    - 2 Tensor Cores (= total 128 multipliers)
        - 2 Octet per 1 Tensor Core
            - 2 Thread Group per 1 Octet
                - 4 x { Four-element dot-product }

|  | Inner-product Tensor Core | Outer-product Tensor Core |
| --- | --- | --- |
| 1TC | 4x4x4 (1 cycle) | 8x8x1 (1 cycle) |
| 2TC | 8x8x4 (2 cycle);                                                           machine level typical m8n8k4 instruction | 8x16x1 (1 cycle);                                                   new machine level HMMA.OHMMA.8161 instruction |
| 2TC | 8x8x16 (8 cycle) | 8x16x16 (16 cycle) |
| 2TC | 16x16x16 (32cycle);                                                     warp level WMMA API | 16x16x16 (32 cycle)                                              new wrap level OWMMA API |

Above is Dense version description

Now, let’s go to sparse version description

## Dual-side Sparse Tensor Core

1. Warp-level interface : SpWMMA API
- wrap-level API

| Warp-level API | machine-level |
| --- | --- |
| 16 sets of  [32x32x1] outer product  | (for one set) 8 instructions of [8x16x1] outer product (= 8 * OHMMA) |
- Skipping Prediction via POPC

![image.png](image%208.png)

Among 8 instruction from one Wrap-level API, some of them would be skipped via population count operations. By counting the 1 in bitmaps, it can determine which OHMMA instruction needs to be skipped.

1. Accumulation Buffer

![image.png](image%209.png)

(a) dense (16 total) , (b) presumably 50% and 50% sparsity (16 total out of 64 → bank conflict)

![image.png](image%2010.png)

For accumulating partial outputs of outer product, there is a gather/scatter operation from sparsity processing. 

The operand collector should combine non-conflicting memory accesses to increase the bandwidth. 

[After Meeting 25.08.26](https://www.notion.so/After-Meeting-25-08-26-25b53787eabe80e2b20fd180f4c2ce87?pvs=21)

# Evaluation

- Does not affect model’s accuracy
- Sparse Tensor Core targetted fixed pruning ratio(75%)
- Proposed work can exploits input and weight sparsity at diverse sparsity level outperforming CUBLASS and cuSparse if higher sparsity than 25%
- For the CNN-based networks, the input and weight sparsity are way higher than 25%. So, proposed work outperforms cuDNN and Sparse Tensor Core.
- For BERT-based network, there is no input sparsity. But, very higher sparsity on weight achieves higher performance.
- Hardware overhead : the author does not reveal how many banks are used and the size of the bank.

# What We(ORCAS) Can do

- RISC-V based solution & open source : this work’s implementation is based on NVIDIA GPU (TC)
- verilog code support : this work’s implementation is based on Accel-Sim
    - How actually implemented in micro-architectural view
- Any new idea to out-perform this work.
    - ratio of { MMA(Tensor Core) instruction / non-MMA instruction} in the kernel of this work. (higher the better)
        - (My GUESS) too many non-compute instruction for one compute-instruction. In short, is Tensor Core active at all time?
        - compute instruction (at Tensor Core) = OHMMA
        - non-compute instruction (at Cuda Core) = (before OHMMA) ‘producing bitmap & encoded feature map at GMEM’, ‘apply mask’, ‘shift left’, ‘pop count’, (after OHMMA) scatter-control (warp bit-map, etc)
        - Or, warp-scheduler would naturally resolve this issue (???, My GUESS is no! data dependency)

# Questions

- ~~In Fig 8, TM and TN in Fig 8 are 32 and 32 each? The paragraph tells that “each thread block computes a 32x32x16 matrix multiplication”.~~ yes
- which units and when does the encoding of bitmaps and densed matrix production happen? And is this process included in the evaluation? Or do the author assume that bitmap production is hided during GEMM op?
    - ~~im2col is done implicitly for feature map. But, even if it is implicit, address calculation and access time should be included~~ (seems included)
    - ~~At least, they have to make bit-map.~~ (seems included)
- ~~Is the original matrix stored in shared mem? And is condensed matrix stored in private register (duplicated)?~~ at GMEM, and reigster yes.
- how the corresponding bitmap data(from certain register) are provided to the accumulation buffer according to the specific OHMMA instruction?
- And how many banks are implemented per accumulation buffer?
