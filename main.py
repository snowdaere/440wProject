import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def get_unique(list):
    # returns list of unique values in a list
    unique = []

    for number in list:
        if number in unique:
            continue
        else:
            unique.append(number)
    return unique

def rangefind(array):
    # takes in array, returns difference between min and max
    return np.nanmax(array) - np.nanmin(array)

class HoboData:
    def __init__(self, filename, name, maxrow):
        [self.lat, self.lon] = np.loadtxt(filename, skiprows=1, max_rows=1, usecols=(0, 1), dtype=str, delimiter=',')
        self.indic = np.loadtxt(filename, delimiter=',', dtype=str, usecols=0, skiprows=3, max_rows=maxrow)
        self.dates = np.loadtxt(filename, delimiter=',', dtype=str, usecols=1, skiprows=3, max_rows=maxrow)
        self.days = np.array([line[3:5] for line in self.dates])
        self.uniquedays = get_unique([line[3:5] for line in self.dates])
        self.times = np.loadtxt(filename, delimiter=',', dtype=str, usecols=2, skiprows=3, max_rows=maxrow)

        self.temps = np.loadtxt(filename, delimiter=',', dtype=float, usecols=3, skiprows=3, max_rows=maxrow)
        self.luxes = np.loadtxt(filename, delimiter=',', dtype=float, usecols=4, skiprows=3, max_rows=maxrow)

        self.name = name

        measurementtime = 5     # five minutes between measurements
        self.dailyluxaverages = np.nanmean(np.reshape(self.luxes, (len(self.uniquedays), int(60 * 24 / measurementtime))), axis=0)
        self.dailytempaverages = np.nanmean(np.reshape(self.temps, (len(self.uniquedays), int(60 * 24 / measurementtime))), axis=0)
        self.dailytimes = self.times[0:288]


        # return average lux for a day (in a month, but whatever)
        # perform the mean without zero values (to avoid the nighttime measurements
        self.dailymeanlux = [0.0] * len(self.uniquedays)
        self.dailymeantemp = [0.0] * len(self.uniquedays)

        for i, day in enumerate(self.uniquedays):
            # calculate mean for a certain day
            luxes = self.luxes[self.days == day]
            temps = self.temps[self.days == day]
            # ignore the zero values
            self.dailymeanlux[i] = np.nanmean(luxes[luxes != 0])
            self.dailymeantemp[i] = np.nanmean(temps[temps != 0])

    def dayofluxdata(self, day):
        return self.luxes[day*288:(day+1)*288]

    def dayoftempdata(self, day):
        return self.temps[day*288:(day+1)*288]

    def dailymeanluxplot(self, axis, color='blue'):
        axis.plot(self.dailytimes, self.dailyluxaverages, label=self.name, color=color)

    def monthlyluxplot(self, axis, color='blue'):
        axis.plot(self.uniquedays, self.dailymeanlux, label=self.name, color=color)

    def dailymeantempplot(self, axis, color='blue'):
        axis.plot(self.dailytimes, self.dailytempaverages, label=self.name, color=color)

    def monthlytempplot(self, axis, color='blue'):
        axis.plot(self.uniquedays, self.dailymeantemp, label=self.name, color=color)

    def luxesplot(self, axes, color='blue'):
        axes.plot(self.indic, self.luxes, color=color, label=self.name)

        xticks = axes.get_xticks()
        axes.set_xticks(xticks[::len(xticks) // 20])  # set new tick positions, in my case marking out each day
        axes.set_xticklabels(self.dates[::len(self.dates) // 20])
        axes.tick_params(axis='x', rotation=90)  # set tick rotation
        axes.margins(x=0)  # set tight margins

    def tempsplot(self, axes, color='blue'):
        axes.plot(self.indic, self.temps, color=color, label=self.name)

        xticks = axes.get_xticks()
        axes.set_xticks(xticks[::len(xticks) // 20])  # set new tick positions, in my case marking out each day
        axes.set_xticklabels(self.dates[::len(self.dates) // 20])
        axes.tick_params(axis='x', rotation=90)  # set tick rotation
        axes.margins(x=0)  # set tight margins


class CloudData:
    def __init__(self, filename) -> None:
        self.dates = np.loadtxt(filename, delimiter=',', dtype=str, usecols=1, skiprows=1)
        self.days = np.array([line[8:10] for line in self.dates])
        self.uniquedays = get_unique([line[8:10] for line in self.dates])
        self.times = np.loadtxt(filename, delimiter=',', dtype=str, usecols=2, skiprows=1)
        self.hours = np.array([line[0:2] for line in self.times])

        # define the lookup for sky cover using the okta definitions
        # For example, since BKN means 5-7 oktas are covered, 6/8 = 0.75 cover
        coverage = {
            '': np.NaN,
            'CLR': .0,
            'FEW': .18,
            'SCT': .44,
            'BKN': .75,
            'OVC': 1.
        }

        # map the function over all
        skyCover = np.loadtxt(filename, delimiter=',', dtype=str, usecols=[3,4,5], skiprows=1)
        skyCover = np.vectorize(coverage.__getitem__)(skyCover)
        # self.skyc1 = skyCover[:, 0]
        # self.skyc2 = skyCover[:, 1]
        # self.skyc3 = skyCover[:, 2]
        self.maxskycover = np.nanmax(skyCover, 1)
    
        # create a list by day of each report
        # take average over valid days
                # return average lux for a day (in a month, but whatever)
        # perform the mean without zero values (to avoid the nighttime measurements
        self.dailymeancoverage = [0.0] * len(self.uniquedays)

        validhours = ['07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18']

        for i, day in enumerate(self.uniquedays):
            # find values with the day and are also valid hours
            coverage = self.maxskycover[np.logical_and(self.days == day, np.isin(self.hours, validhours))]

            # take the mean
            self.dailymeancoverage[i] = np.nanmean(coverage)




class ZenithData:
    pass




if __name__ == '__main__':
    Datafiles = ['Data/DenseForest.csv',
                 'Data/LightForest.csv',
                 'Data/ParkingMeadow.csv',
                 'Data/StuckemanParkingLot.csv',
                 'Data/StuckemanWoods.csv',
                 'Data/WildflowerMeadow.csv']

    Names = ['Parking Forest', 'Meadow Forest', 'Parking Meadow', 'Hort Lot', 'Hort Woods', 'Meadow Field']

    # load that. row 5921 is midnight the day before I took the sensors down
    datasets = [ParkingForest, MeadowForest, ParkingField, HortLot, HortWoods, MeadowField] = \
        [HoboData(file, Names[i], maxrow=5760) for i, file in enumerate(Datafiles)]
    
    clouds = CloudData('Data/UNV.csv')

    

    def factor():
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), layout='tight')
        
        # create a bigx and bigy, where x is all the unshaded data, and y is all the shaded data

        bigx = np.array([item for sublist in [list(i.dailymeanlux) for i in [ParkingField, HortLot, MeadowField]] for item in sublist])
        bigy = np.array([item for sublist in [list(i.dailymeanlux) for i in [ParkingForest, HortWoods, MeadowForest]] for item in sublist])

        # convert to numpy-usable stuff
        bigx = bigx[:, np.newaxis]
        bigy = bigy[:, np.newaxis]

        # plot
        ax.scatter(bigx, bigy)
        ax.set_xlabel('Unshaded Luminosity (lux)')
        ax.set_ylabel('Shaded Luminosity (lux)')
        ax.set_title('Relationship of Unshaded and Shaded Lum.')

        # do proportional fit
        shadedfit, _, _, _ = np.linalg.lstsq(bigx, bigy)
        ax.plot(bigx, shadedfit*bigx, label='Trend', color='black')

        fig.savefig('Bigplot.png')

        residuals = (bigy - shadedfit*bigx)/rangefind(bigy)

        # plot the residuals
        fig, ax = plt.subplots(1, 1, figsize=(4, 3), layout='tight')
        ax.set_title('Residuals, Unshaded and Shaded Lum.')
        ax.hist(residuals, density=True, bins=20)
        ax.set_xlabel('Error/Y Range (lux/lux)')
        ax.set_ylabel('Density')

        print(f'residuals mean: {np.nanmean(residuals)}')
        print(f'residuals std: {np.nanstd(residuals)}')

        fig.savefig('bigresiduals.png')

    def lux3():
        # plot daily means over time
        fig, (ax1, ax2) = plt.subplots(2, 1, layout='tight', figsize=(8, 8), sharex=True)
        fig.suptitle('Mean Luminosity over October')
        ax1.set_title('Shaded')
        ax2.set_title('Unshaded')

        HortLot.monthlyluxplot(ax2, color='orange')
        HortWoods.monthlyluxplot(ax1, color='tomato')
        ParkingField.monthlyluxplot(ax2, color='limegreen')
        ParkingForest.monthlyluxplot(ax1, color='gold')
        MeadowField.monthlyluxplot(ax2, color='darkorchid')
        MeadowForest.monthlyluxplot(ax1, color='cornflowerblue')
        # get days for general use
        days = HortWoods.uniquedays

        ax1.set_ylabel('Luminosity (lux)')
        ax2.set_ylabel('Luminosity (lux)')
        ax2.set_xlabel('Day')



        # for both, find mean points, and line of best fit
        unshaded = np.nanmean(
            np.column_stack((HortLot.dailymeanlux, ParkingField.dailymeanlux, MeadowField.dailymeanlux)), axis=1)
        shaded = np.nanmean(
            np.column_stack((HortWoods.dailymeanlux, ParkingForest.dailymeanlux, MeadowForest.dailymeanlux)), axis=1)
        ax1.scatter(days, shaded, label='Mean', color='black')
        ax2.scatter(days, unshaded, label='Mean', color='black')

        dayvalues = [float(day) for day in days]

        # now find and plot line of best fit
        shadedfit = np.polyfit(dayvalues, shaded, 1)
        ax1.plot(days, np.polyval(shadedfit, dayvalues), label='Trend', color='black')
        unshadedfit = np.polyfit(dayvalues, unshaded, 1)
        ax2.plot(days, np.polyval(unshadedfit, dayvalues), label='Trend', color='black')

        # print equations and r2 values
        # perform statistical analyses

        # then calculate extinction coefficients and shit
        # account for clouds
        # try to process out the noise using the unshaded data

        ax1.legend()
        ax2.legend()


        fig.savefig('MonthlyPlot.png')

        # plot residuals
        shadedresiduals = (shaded-np.polyval(shadedfit, dayvalues))/rangefind(shaded)
        unshadedresiduals = (unshaded-np.polyval(unshadedfit, dayvalues))/rangefind(unshaded)

        fig, (ax1, ax2) = plt.subplots(1, 2, layout='tight', figsize=(10, 4), sharey=False)
        ax1.set_title('Residuals, Shaded Lum.')
        ax1.hist(shadedresiduals, density=True, bins=10)
        ax1.set_xlabel('Error/Data Range (lux/lux)')
        ax1.set_ylabel('Density')

        ax2.set_title('Residuals, Unshaded Lum.')
        ax2.hist(unshadedresiduals, density=True, bins=10)
        ax2.set_xlabel('Error/Data Range (lux/lux)')
        ax2.set_ylabel('Density')

        fig.savefig('trendresiduals.png')

        print(f'shaded residuals mean: {np.nanmean(shadedresiduals)}')
        print(f'shaded residuals std: {np.nanstd(shadedresiduals)}')

        print(f'unshaded residuals mean: {np.nanmean(unshadedresiduals)}')
        print(f'unshaded residuals std: {np.nanstd(unshadedresiduals)}')

    def coverageplot():
        # create a scatter plot of daily average cloud coverage during the day against daily average luminosity
        fig, (ax1, ax2) = plt.subplots(1, 2, layout='tight', figsize=(10, 4), sharey=False)

        

        unshaded = np.nanmean(
            np.column_stack((HortLot.dailymeanlux, ParkingField.dailymeanlux, MeadowField.dailymeanlux)), axis=1)
        shaded = np.nanmean(
            np.column_stack((HortWoods.dailymeanlux, ParkingForest.dailymeanlux, MeadowForest.dailymeanlux)), axis=1)
        
        # take out the last two days of coverage data; those days werent studied
        ax1.scatter(clouds.dailymeancoverage[:-1], shaded, color='cornflowerblue')
        ax2.scatter(clouds.dailymeancoverage[:-1], unshaded, color='cornflowerblue')

        # label each point with its date
        [ax1.text(x=clouds.dailymeancoverage[i], y=shaded[i], s=clouds.uniquedays[i]) for i in range(0, 20)]        
        [ax2.text(x=clouds.dailymeancoverage[i], y=unshaded[i], s=clouds.uniquedays[i]) for i in range(0, 20)]

        # fit linear trends
        # now find and plot line of best fit
        shadedfit = np.polyfit(clouds.dailymeancoverage[:-1], shaded, 1)
        ax1.plot(clouds.dailymeancoverage[:-1], np.polyval(shadedfit, clouds.dailymeancoverage[:-1]), label='Trend', color='black')
        unshadedfit = np.polyfit(clouds.dailymeancoverage[:-1], unshaded, 1)
        ax2.plot(clouds.dailymeancoverage[:-1], np.polyval(unshadedfit, clouds.dailymeancoverage[:-1]), label='Trend', color='black')

        # print trendline numbers
        print(f'shaded fit: {shadedfit}')

        print(f'unshaded fit: {unshadedfit}')

        fig.suptitle('Daily Mean Sky Cover v Lum.')
        ax1.set_xlabel('Sky Cover (frac OVC)')
        ax2.set_xlabel('Sky Cover (frac OVC)')
        ax1.set_title('Shaded')
        ax2.set_title('Unshaded')
        ax1.set_ylabel('Luminosity (lux)')
        fig.savefig('CloudCoverage.png')


        # plot residuals
        shadedresiduals = (shaded-np.polyval(shadedfit, clouds.dailymeancoverage[:-1]))/rangefind(shaded)
        unshadedresiduals = (unshaded-np.polyval(unshadedfit, clouds.dailymeancoverage[:-1]))/rangefind(unshaded)

        fig, (ax1, ax2) = plt.subplots(1, 2, layout='tight', figsize=(10, 4), sharey=False)
        ax1.set_title('Residuals, Shaded')
        ax1.hist(shadedresiduals, density=True, bins=10)
        ax1.set_xlabel('Error/Data Range (lux/lux)')
        ax1.set_ylabel('Density')

        ax2.set_title('Residuals, Unshaded')
        ax2.hist(unshadedresiduals, density=True, bins=10)
        ax2.set_xlabel('Error/Data Range (lux/lux)')
        ax2.set_ylabel('Density')

        fig.savefig('cloudresiduals.png')

        print(f'shaded residuals mean: {np.nanmean(shadedresiduals)}')
        print(f'shaded residuals std: {np.nanstd(shadedresiduals)}')

        print(f'unshaded residuals mean: {np.nanmean(unshadedresiduals)}')
        print(f'unshaded residuals std: {np.nanstd(unshadedresiduals)}')

        
    def dayplot(day):
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8.5, 11), layout='tight')
        fig.suptitle(f'Luminosity comparison for October {day+1} 2022')
        labels = HortLot.dailytimes

        # plot Hort woods data
        ax1.set_title('Hort Woods')
        ax1.plot(labels, HortLot.dayofluxdata(day), color='orange', label='Hort Lot')
        ax1.plot(labels, HortWoods.dayofluxdata(day), color='tomato', label='Hort Woods')
        ax1.legend()
        ax1.set_ylabel('Luminosity (lux)')


        # plot parking data
        ax2.set_title('Parking Area')
        ax2.plot(labels, ParkingField.dayofluxdata(day), color='limegreen', label='Parking Field')
        ax2.plot(labels, ParkingForest.dayofluxdata(day), color='gold', label='Parking Forest')
        ax2.legend()
        ax2.set_ylabel('Luminosity (lux)')


        # plot meadow data
        ax3.set_title('Meadow Area')
        ax3.plot(labels, MeadowField.dayofluxdata(day), color='darkorchid', label='Meadow Field')
        ax3.plot(labels, MeadowForest.dayofluxdata(day), color='cornflowerblue', label='Meadow Forest')
        ax3.legend()
        ax3.set_xlabel('Time of Day')
        ax3.set_ylabel('Luminosity (lux)')

        for ax in [ax1, ax2, ax3]:
            ax.set_xticks(labels[::len(labels) // 12])  # set new tick positions, in my case marking out every 2 hours
            ax.set_xticklabels(labels[::len(labels) // 12])
            ax.tick_params(axis='x', rotation=60)  # set tick rotation
            ax.margins(x=0)  # set tight margins


        fig.savefig(f'ComparisonDay{day+1}.png')
    
    def daytempplot(day):
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8.5, 11), layout='tight')
        fig.suptitle(f'Luminosity comparison for October {day+1} 2022')
        labels = HortLot.dailytimes

        # plot Hort woods data
        ax1.set_title('Hort Woods')
        ax1.plot(labels, HortLot.dayoftempdata(day), color='orange', label='Hort Lot')
        ax1.plot(labels, HortWoods.dayoftempdata(day), color='tomato', label='Hort Woods')


        # plot parking data
        ax2.set_title('Parking Area')
        ax2.plot(labels, ParkingField.dayoftempdata(day), color='limegreen', label='Parking Field')
        ax2.plot(labels, ParkingForest.dayoftempdata(day), color='gold', label='Parking Forest')


        # plot meadow data
        ax3.set_title('Meadow Area')
        ax3.plot(labels, MeadowField.dayoftempdata(day), color='darkorchid', label='Meadow Field')
        ax3.plot(labels, MeadowForest.dayoftempdata(day), color='cornflowerblue', label='Meadow Forest')
        ax3.set_xlabel('Time of Day')

        for ax in [ax1, ax2, ax3]:
            ax.set_xticks(labels[::len(labels) // 12])  # set new tick positions, in my case marking out every 2 hours
            ax.set_xticklabels(labels[::len(labels) // 12])
            ax.tick_params(axis='x', rotation=60)  # set tick rotation
            ax.margins(x=0)  # set tight margins
            ax.set_ylabel('Temperature (C)')
            ax.legend()


        fig.savefig(f'tempComparison.png')
    

    def yfplot():
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharey = True, sharex=True, figsize=(8.5, 11), layout='tight')

        F0 = [HortLot, ParkingField, MeadowField]
        F = [HortWoods, ParkingForest, MeadowForest]
        colors = ['orange', 'limegreen', 'darkorchid']
        yfs = [[], [], []]

        for i, ax in enumerate([ax1, ax2, ax3]):
            yf = -1* np.log(np.divide(F0[i].dailymeanlux, F[i].dailymeanlux))
            yfs[i] = yf
            days = HortLot.uniquedays
            ax.set_ylabel('yf (unitless)')
            ax.scatter(days, yf, color=colors[i])

            dayvalues = [float(day) for day in days]

            # plot best fit line
            unshadedfit = np.polyfit(dayvalues, yf, 1)
            ax.plot(days, np.polyval(unshadedfit, dayvalues), label='Trend', color='black')
            ax.text(1, -1.25, f'y = mx + b\nm = {unshadedfit[0]:.3} /day\nb = {unshadedfit[1]:.3}')

        fig.suptitle('Daily Mean Values of yf')
        ax1.set_title('Hort Woods')
        ax2.set_title('Parking Area')
        ax3.set_title('Meadow Area')
        ax3.set_xlabel('Day of October')

        fig.savefig('YFplot.png')

        
        
        
        #### NEXT PLOT

        fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, sharex = True, sharey=True, figsize=(8.5, 4), layout='tight')

        F0 = [HortLot, ParkingField, MeadowField]
        F = [HortWoods, ParkingForest, MeadowForest]
        colors = ['orange', 'limegreen', 'darkorchid']
        yfs = [[], [], []]
        yfsresiduals = [[], [], []]


        for i, ax in enumerate([ax1, ax2, ax3]):
            yf = -1* np.log(np.divide(F0[i].dailymeanlux, F[i].dailymeanlux))
            yfs[i] = yf
            days = HortLot.uniquedays
            unshadedfit = np.polyfit(dayvalues, yf, 1)

            residuals = yf-np.polyval(unshadedfit, dayvalues)
            yfsresiduals[i] = residuals

            ax.set_xlabel('yf (unitless)')
            ax.text(-0.5, 4.5, f'Mean: {np.mean(residuals):.3}\nSTD: {np.std(residuals):.3}')

            ax.hist(residuals, density=False, color=colors[i])

        fig.suptitle('Distributions of yf Residuals')
        ax1.set_title('Hort Woods')
        ax2.set_title('Parking Area')
        ax3.set_title('Meadow Area')
        ax1.set_ylabel('n Occurences')


        fig.savefig('YFhistResiduals.png')

        # test plot
        fig, ax = plt.subplots(layout = 'tight', figsize=(4, 4))
        ax.scatter(yfsresiduals[0], yfsresiduals[1])
        fig.savefig('isthiscorrelation.png')
        ax.set_ylabel('Meadow yf resid.')
        ax.set_xlabel('Parking yf resid.')
        ax.set_title('Correlation of yf for Meadow and Parking Datasets')
        fig.savefig('isthiscorrelation.png')
        print(pearsonr(yfsresiduals[0], yfsresiduals[1]))

        


    yfplot()

    def FF0():
        fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8.5, 11), layout='tight')

        F0 = [HortLot, ParkingField, MeadowField]
        F = [HortWoods, ParkingForest, MeadowForest]
        colors = ['orange', 'limegreen', 'darkorchid']
        colors1 = ['tomato', 'gold', 'cornflowerblue']

        for i, ax in enumerate([ax1, ax2, ax3]):
            ax.set_ylabel('Mean Luminosity (lux)')
            ax.scatter(HortLot.uniquedays, F0[i].dailymeanlux, color=colors[i], label='F0')
            ax.scatter(HortLot.uniquedays, F[i].dailymeanlux, color=colors1[i], label='F')
            ax.legend()

        fig.suptitle('Daily Mean Values of F0 and F')
        ax1.set_title('Hort Woods')
        ax2.set_title('Parking Area')
        ax3.set_title('Meadow Area')
        ax3.set_xlabel('Day of October')

        fig.savefig('FF0.png')
