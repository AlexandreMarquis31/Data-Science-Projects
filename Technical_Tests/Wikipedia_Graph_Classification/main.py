import optparse
from wktools.wkpipeline import Pipeline, DataLoader, GraphConstructor, ClusterConstructor, Observer


def main():
    # For the command line arguments
    parser = optparse.OptionParser()
    parser.add_option("-n", "--n", help="Number of common tokens to link pages ")
    parser.add_option("-f", "--file", help='File path')
    opts, args = parser.parse_args()
    if opts.file is None:
        parser.error('Filepath required')
    n = opts.n if opts.n is not None else 400

    # The pipeline
    pipeline = Pipeline([
        DataLoader(),
        (GraphConstructor(), n),
        ClusterConstructor(),
        Observer()
    ])
    pipeline.run(opts.file)


if __name__ == "__main__":
    main()
