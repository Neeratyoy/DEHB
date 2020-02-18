

def create_plot(plt, methods, path, regret_type, fill_trajectory,
                colors, linestyles, marker, n_runs=500, limit=1e7):

    # plot limits
    min_time = np.inf
    max_time = 0
    min_regret = 1
    max_regret = 0

    no_runs_found = False
    # looping and plotting for all methods
    for index, (m, label) in enumerate(methods):

        regret = []
        runtimes = []
        for k, i in enumerate(np.arange(n_runs)):
            try:
                res = json.load(open(os.path.join(path, m, "run_%d.json" % i)))
                no_runs_found = False
            except Exception as e:
                print(m, i, e)
                no_runs_found = True
                continue
            regret_key =  "regret_validation" if regret_type == 'validation' else "regret_test"
            runtime_key = "runtime"
            _, idx = np.unique(res[regret_key], return_index=True)
            idx.sort()
            regret.append(np.array(res[regret_key])[idx])
            runtimes.append(np.array(res[runtime_key])[idx])

        if not no_runs_found:
            # finds the latest time where the first measurement was made across runs
            t = np.max([runtimes[i][0] for i in range(len(runtimes))])
            min_time = min(min_time, t)
            te, time = fill_trajectory(regret, runtimes, replace_nan=1)

            idx = time.tolist().index(t)
            te = te[idx:, :]
            time = time[idx:]

            # Clips off all measurements after 10^7s
            idx = np.where(time < limit)[0]

            print("{}. Plotting for {}".format(index, m))
            print(len(regret), len(runtimes))
            # The mean plot
            plt.plot(time[idx], np.mean(te, axis=1)[idx], color=colors[index],
                     linewidth=4, label=label, linestyle=linestyles[index % len(linestyles)],
                     marker=marker[index % len(marker)], markevery=(0.1,0.1), markersize=15)
            # The error band
            plt.fill_between(time[idx],
                             np.mean(te, axis=1)[idx] + 2 * stats.sem(te[idx], axis=1),
                             np.mean(te[idx], axis=1)[idx] - 2 * stats.sem(te[idx], axis=1),
                             color="C%d" % index, alpha=0.2)

            # Stats to dynamically impose limits on the axes of the plots
            max_time = max(max_time, time[idx][-1])
            min_regret = min(min_regret, np.mean(te, axis=1)[idx][-1])
            max_regret = max(max_regret, np.mean(te, axis=1)[idx][0])

    return plt, min_time, max_time, min_regret, max_regret