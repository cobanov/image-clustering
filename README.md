# Image Clustering

## Installation

For CPU Installation

```bash
pip install -r requirements.txt
```

For GPU installlation

```bash
pip install -r requirements_gpu.txt
```

## Demo

![animals](assets/animals.png)
![flowers](assets/flowers.png)

## Usage

```bash
usage: main.py [-h] [-i INPUT] [-c CLUSTER] [-p PCA]

Image caption CLI

optional arguments:
  -h, --help                        show this help message and exit
  -i INPUT, --input INPUT           Input directory path, such as ./images
  -c CLUSTER, --cluster CLUSTER     How many cluster will be
  -p PCA, --pca PCA                 PCA Dimensions
  --cpu                             Run on CPU
```

### Example

For CPU

```bash
python main.py -i "D:\New folder\random_1500" -c 5 -p 16 --cpu
```

For GPU

```bash
python main.py -i "D:\New folder\random_1500" -c 5 -p 16
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

[MIT](https://choosealicense.com/licenses/mit/)
