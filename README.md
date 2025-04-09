
<p align="left">
  <img src="https://raw.githubusercontent.com/amesch441/iSCALE/main/assets/iSCALE_logo2.png" width="250"/>
</p>


## Scaling up spatial transcriptomics for large-sized tissues: uncovering cellular-level tissue architecture beyond conventional platforms with iSCALE

This software package implements iSCALE
(Inferring Spatially resolved Cellular Architectures for Large-sized tissue Environments),
A novel framework designed to integrate multiple daughter captures and utilize H&E information from large tissue samples, enabling the prediction of gene expression in large-sized tissues with near single-cell resolution.


<p align="center">
  <img src="https://raw.githubusercontent.com/amesch441/iSCALE/main/assets/iSCALE_workflow2.png" width="1000"/>
</p>
<p align="center">
  <strong>Figure:</strong> <em>iSCALE workflow</em>
</p>



# Get Started

Please download checkpoints.zip and demo.zip folders from: https://upenn.box.com/s/cburekr425ibu276wyxki09q35z2o3x0
Unzip folders and place into your iSCALE directory.

To run the demo, use the file run_iSCALE.sh with directory prefix="Data/demo/gastricTumor/". The ground truth for this gastric tumor tissue can be found in the cnts-truth-agg folder. 

```
sbatch run_iSCALE.sh 
```

Using GPUs is highly recommended.

Additional information about data input structure and parameters will be provided soon
