# Redundant Structured Light
Github Repo for Optics Express 2024 Publication: "Robust Structured Light with Efficient Redundant Codes"

:book: Paper:
https://opg.optica.org/oe/fulltext.cfm?uri=oe-32-19-33507&id=558446

:star: Youtube video: 
https://youtu.be/9wjhcGQSmaA

**Update 12/22/2024:**

We uploaded the scripts and demo data. Please try it out if you are interested!

In `demo.ipynb` notebook, we include:

1. Simulated evaluation for data stored in `data` folder

2. Real-world evaluation for data stored at https://drive.google.com/drive/folders/1DkGQb063Rrq0oCNld_e-dtgbfNXwphMf?usp=drive_link

In `MRF` folder, we incldue the scripts for Markov-Random-Field based spatial context decoder.

In `sl_coding_denoising` folder, we include the scripts for Convolution-Neural-Network based spatial context decoder.

Otherwise, the basic ZNCC decoder is used (https://openaccess.thecvf.com/content_cvpr_2018/html/Mirdehghan_Optimal_Structured_Light_CVPR_2018_paper.html).

## Citation
If you find our work useful in your research, please consider citing:

        @article{sun2024robust,
          title={Robust structured light with efficient redundant codes},
          author={Sun, Zhanghao and Zuo, Xinxin and Huo, Dong and Zhang, Yu and Qian, Yiming and Wang, Jian},
          journal={Optics Express},
          volume={32},
          number={19},
          pages={33507--33520},
          year={2024},
          publisher={Optica Publishing Group}
        }
