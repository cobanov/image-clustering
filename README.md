# Image Clustering

## Installation

```bash
pip install -r requirements.txt
```

- torch
- img2vec
- numpy
- scikit-learn
- matplotlib

## Demo

![animals](assets/animals.png)
![flowers](assets/flowers.png)

## Usage

```bash
usage: inference.py [-h] [-i INPUT] [-b BATCH] [-p PATHS] [-g GPU_ID]        

Image caption CLI

optional arguments:
  -h, --help                      show this help message and exit
  -i INPUT,  --input INPUT        Input directoryt path, such as ./images
  -b BATCH,  --batch BATCH        Batch size
  -p PATHS,  --paths PATHS        A any.txt files contains all image paths.
  -g GPU_ID, --gpu-id GPU_ID      gpu device to use (default=0) can be 0,1,2 for multi-gpu
```

### Example

```bash
python main.py -i /path/images/folder --batch 8 --gpu 0 
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
