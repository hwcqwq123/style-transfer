# Style Transfer Project
这个readme由ai生成啊，只要我这句是人话

基于 Vue 3 + Flask 的图像风格迁移系统，支持上传内容图和风格图，选择不同算法后生成结果图。

## 项目简介

本项目实现了一个前后端分离的图像风格迁移网页系统，主要功能包括：

- 上传内容图
- 上传风格图
- 选择生成算法
- 展示生成结果
- 下载结果图

当前支持的算法包括：

- Adam
- LBFGS
- CycleGAN

其中：

- Adam 和 LBFGS 用于经典 Gatys 风格迁移
- CycleGAN 用于基于已训练模型的图像域转换

---

## 项目结构

```bash
style-transfer/
├─ frontend/          # Vue 前端
│  ├─ src/
│  ├─ package.json
│  └─ ...
│
├─ backend/           # Flask 后端
│  ├─ app.py
│  ├─ requirements.txt
│  ├─ services/
│  ├─ uploads/
│  ├─ checkpoints/
│  └─ pytorch-CycleGAN-and-pix2pix/
│
└─ README.md
