import matplotlib.pyplot as plt
from sklearn import datasets as ds
from sklearn import cluster


def main():
    inertia_list = []
    k_values = []
    X = ds.load_iris().data
    for k in range(1, 11):
        inertia = cluster.KMeans(n_clusters=k, init='k-means++', random_state=0).fit(X).inertia_
        inertia_list.append(inertia)
        k_values.append(k)
    plt.plot(k_values, inertia_list)
    plt.title('Elbow Method for selection of optimal "K" clusters',
              fontdict={'family': 'DejaVu Sans',
                        'color': 'brown',
                        'weight': 'bold',
                        'size': 15,
                        })
    plt.xlabel('K', fontdict={'size': 14,
                              'weight': 'bold'})
    plt.ylabel('Inertia', fontdict={'size': 14,
                                    'weight': 'bold'})
    plt.locator_params('x', nbins=10)
    axes = plt.gca()
    axes.plot(3, 90, 'o', ms=25, mec='black', mfc='none', mew=3)
    axes.annotate('Elbow Point', weight="bold", xy=(3, 90), xytext=(5, 190), color='black', size='large',
                  arrowprops=dict(arrowstyle='fancy,tail_width=0.2,head_width=1,head_length=0.8', facecolor='black',
                                  shrinkB=15))

    plt.savefig("elbow.png")


if __name__ == "__main__":
    main()
