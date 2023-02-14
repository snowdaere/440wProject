def lux1():
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8.5, 11), layout='tight')
    fig.suptitle('Luminosity over Oct. 1 to Oct. 20')

    # plot Hort woods data
    ax1.set_title('Hort Woods')
    HortLot.luxesplot(ax1, color='orange')
    HortWoods.luxesplot(ax1, color='tomato')
    ax1.legend()
    ax1.set_ylabel('Luminosity (lux)')


    # plot parking data
    ax2.set_title('Parking Area')
    ParkingField.luxesplot(ax2, color='limegreen')
    ParkingForest.luxesplot(ax2, color='gold')
    ax2.legend()
    ax3.set_ylabel('Luminosity (lux)')


    # plot meadow data
    ax3.set_title('Meadow Area')
    MeadowField.luxesplot(ax3, color='darkorchid')
    MeadowForest.luxesplot(ax3, color='cornflowerblue')
    ax3.legend()
    ax3.set_xlabel('Day of the Month')
    ax3.set_ylabel('Luminosity (lux)')


    fig.savefig('LuminosityPlot.png')


def lux2():
    # plot the daily means for each site out
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), layout='tight')

    HortLot.dailymeanluxplot(ax, color='orange')
    HortWoods.dailymeanluxplot(ax, color='tomato')
    ParkingField.dailymeanluxplot(ax, color='limegreen')
    ParkingForest.dailymeanluxplot(ax, color='gold')
    MeadowField.dailymeanluxplot(ax, color='darkorchid')
    MeadowForest.dailymeanluxplot(ax, color='cornflowerblue')

    # get some time labels for plotting
    labels = HortLot.dailytimes

    # plot means for shaded and unshaded
    unshaded = np.nanmean(
        np.column_stack((HortLot.dailyluxaverages, ParkingField.dailyluxaverages, MeadowField.dailyluxaverages)), axis=1)
    shaded = np.nanmean(
        np.column_stack((HortWoods.dailyluxaverages, ParkingForest.dailyluxaverages, MeadowForest.dailyluxaverages)), axis=1)
    ax.plot(labels, unshaded, label='Unshaded Mean', color='grey')
    ax.plot(labels, shaded, label='Shaded Mean', color='black')

    # label the x axis with times
    ax.set_xticks(labels[::len(labels) // 12])  # set new tick positions, in my case marking out every 2 hours
    ax.set_xticklabels(labels[::len(labels) // 12])
    ax.tick_params(axis='x', rotation=60)  # set tick rotation
    ax.margins(x=0)  # set tight margins

    # make it a plot
    ax.set_ylabel('Luminosity (lux)')
    ax.set_xlabel('Time of Day')
    ax.set_title('Daily Mean Luminosity')
    ax.legend(loc='upper left')
    fig.savefig('DailyMeanPlot.png')


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


def temp1():
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1, sharex=True, figsize=(8.5, 11), layout='tight')
    fig.suptitle('Temperature over Oct. 1 to Oct. 20')

    # plot Hort woods data
    ax1.set_title('Hort Woods')
    HortLot.tempsplot(ax1, color='orange')
    HortWoods.tempsplot(ax1, color='tomato')
    ax1.legend()
    ax1.set_ylabel('Temperature (C)')


    # plot parking data
    ax2.set_title('Parking Area')
    ParkingField.tempsplot(ax2, color='limegreen')
    ParkingForest.tempsplot(ax2, color='gold')
    ax2.legend()
    ax3.set_ylabel('Temperature (C)')


    # plot meadow data
    ax3.set_title('Meadow Area')
    MeadowField.tempsplot(ax3, color='darkorchid')
    MeadowForest.tempsplot(ax3, color='cornflowerblue')
    ax3.legend()
    ax3.set_xlabel('Day of the Month')
    ax3.set_ylabel('Temperature (C)')


    fig.savefig('TemperaturePlot.png')


def temp2():
    # plot the daily means for each site out
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), layout='tight')

    HortLot.dailymeantempplot(ax, color='orange')
    HortWoods.dailymeantempplot(ax, color='tomato')
    ParkingField.dailymeantempplot(ax, color='limegreen')
    ParkingForest.dailymeantempplot(ax, color='gold')
    MeadowField.dailymeantempplot(ax, color='darkorchid')
    MeadowForest.dailymeantempplot(ax, color='cornflowerblue')

    # get some time labels for plotting
    labels = HortLot.dailytimes

    # plot means for shaded and unshaded
    unshaded = np.nanmean(
        np.column_stack((HortLot.dailytempaverages, ParkingField.dailytempaverages, MeadowField.dailytempaverages)), axis=1)
    shaded = np.nanmean(
        np.column_stack((HortWoods.dailytempaverages, ParkingForest.dailytempaverages, MeadowForest.dailytempaverages)), axis=1)
    ax.plot(labels, unshaded, label='Unshaded Mean', color='grey')
    ax.plot(labels, shaded, label='Shaded Mean', color='black')

    # label the x axis with times
    ax.set_xticks(labels[::len(labels) // 12])  # set new tick positions, in my case marking out every 2 hours
    ax.set_xticklabels(labels[::len(labels) // 12])
    ax.tick_params(axis='x', rotation=60)  # set tick rotation
    ax.margins(x=0)  # set tight margins

    # make it a plot
    ax.set_ylabel('Temperature (C)')
    ax.set_xlabel('Time of Day')
    ax.set_title('Daily Mean Temperature')
    ax.legend(loc='upper left')
    fig.savefig('DailyMeanTemperaturePlot.png')


def temp3():
    # plot daily means over time
    fig, (ax1, ax2) = plt.subplots(2, 1, layout='tight', figsize=(8, 8), sharex=True)
    fig.suptitle('Mean Temperature over October')
    ax1.set_title('Shaded')
    ax2.set_title('Unshaded')

    HortLot.monthlytempplot(ax2, color='orange')
    HortWoods.monthlytempplot(ax1, color='tomato')
    ParkingField.monthlytempplot(ax2, color='limegreen')
    ParkingForest.monthlytempplot(ax1, color='gold')
    MeadowField.monthlytempplot(ax2, color='darkorchid')
    MeadowForest.monthlytempplot(ax1, color='cornflowerblue')
    # get days for general use
    days = HortWoods.uniquedays

    ax1.set_ylabel('Temperature (C)')
    ax2.set_ylabel('Temperature (C)')
    ax2.set_xlabel('Day')



    # for both, find mean points, and line of best fit
    unshaded = np.nanmean(
        np.column_stack((HortLot.dailymeantemp, ParkingField.dailymeantemp, MeadowField.dailymeantemp)), axis=1)
    shaded = np.nanmean(
        np.column_stack((HortWoods.dailymeantemp, ParkingForest.dailymeantemp, MeadowForest.dailymeantemp)), axis=1)
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


    fig.savefig('MonthlyTemperaturePlot.png')


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

        fig.savefig('CloudCoverage.png')

        pass
