import matplotlib.animation as animation
from matplotlib import pyplot as plt
from matplotlib.animation import HTMLWriter
import numpy as np

def p_v_init(num_particles, speed, R, L):

    #initial positions
    ps= np.zeros((num_particles,2))
    spacing = 1.2*2*R
    iters = int(L/spacing)
    count = 0
    flag = 0
    for i in range(iters):
        for j in range(iters):
            ps[count] = [i*spacing,j*spacing]
            count += 1
            if count == num_particles:
                flag = 1
                break
        if flag == 1:
            break
    if flag == 0:
        print('too many particles to fit in system! Decrease N and try again')
        exit()

    ps = ps - L*0.5

    #initialize velocities
    theta = 2*np.pi*np.random.rand(num_particles)
    x_vs = speed*np.cos(theta)
    y_vs = speed*np.sin(theta)
    x_vs = np.reshape(x_vs, (num_particles,1))
    y_vs = np.reshape(y_vs, (num_particles,1))
    vs = np.concatenate((x_vs, y_vs), axis=1)
    return ps, vs

def calc_coll_time(ri, rj, vi, vj, L, D2):
    vij = vi-vj
    rij = (ri-rj) - L*np.round((ri-rj)/L)
    rij2 = np.dot(rij, rij)

    #use quadratic formula to calculate collision time
    bij = np.dot(rij, vij)

    if bij >= 0 : #particles not on correct trajectory to collide
        return np.Inf

    vij2 = np.dot(vij, vij)
    det = bij**2-(vij2)*(rij2-D2)
    vij2 = np.dot(vij, vij)
    if det < 0: #particles will not collide
        return np.Inf
    t_coll = (-bij-(det**.5))/(vij2)
    return t_coll

def calc_coll_times_all(collision_times, ps, vs, num_particles, agg_indices, L, D2):
    for i in range(num_particles):
        for j in range(i+1, num_particles):
            if agg_indices[i] != agg_indices[j]:
                collision_times[i,j] = calc_coll_time(ps[i], ps[j], vs[i], vs[j], L, D2)
    return collision_times

def merge_shift_clusters(agg_indices, agg_array, agg_array_lens,agg_clusters, ind1, ind2):
    min_c = min(agg_indices[ind1],  agg_indices[ind2])
    max_c = max(agg_indices[ind1],  agg_indices[ind2])
    for i in range(len(agg_indices)):
        if agg_indices[i] == max_c:
            agg_indices[i] = min_c
        elif agg_indices[i] > max_c:
            agg_indices[i] = agg_indices[i] - 1
    agg_array[min_c, agg_array_lens[min_c] : agg_array_lens[min_c] + agg_array_lens[max_c]] = agg_array[max_c, 0 : agg_array_lens[max_c]]
    agg_array_lens[min_c] += agg_array_lens[max_c]
    agg_array_lens[max_c] = 0
    for i in range(max_c, agg_clusters-1):
        agg_array[i] = agg_array[i+1]
        agg_array_lens[i] = agg_array_lens[i+1]
    agg_array[agg_clusters-1] = 0
    agg_array_lens[agg_clusters-1] = 0

def collide(ps, vs, pair, t_coll, agg_array, agg_indices, num_agg, agg_clusters, agg_array_lens, L):
    #move everything up to time of collision
    ps = ps + t_coll * vs
    ps -= L*np.around(ps/L)

    v_av = (agg_array_lens[agg_indices[pair[0]]]*vs[pair[0]] + agg_array_lens[agg_indices[pair[1]]]*vs[pair[1]]) \
        / (agg_array_lens[agg_indices[pair[0]]] + agg_array_lens[agg_indices[pair[1]]])

    v_av = np.sqrt((1.0/np.dot(v_av, v_av))) * v_av

    for i in range(agg_array_lens[agg_indices[pair[0]]]):
        vs[agg_array[agg_indices[pair[0]], i]] = v_av
    for i in range(agg_array_lens[agg_indices[pair[1]]]):
        vs[agg_array[agg_indices[pair[1]], i]] = v_av

    #merge clusters, and shift other cluster numbers as necessary
    merge_shift_clusters(agg_indices, agg_array, agg_array_lens, agg_clusters, pair[0], pair[1])
    agg_clusters -= 1

    return ps, vs, num_agg, agg_clusters

def recalc_coll_times(pair, ps, vs, num_particles, ts_next_list, partners_list, agg_indices, L, D2):
    ts_next_list[pair[0]] = np.Inf
    ts_next_list[pair[1]] = np.Inf
    for i in range(num_particles):
        for j in pair:
            if agg_indices[i] != agg_indices[j]:
                t_coll = calc_coll_time(ps[i], ps[j], vs[i], vs[j], L, D2)
                if t_coll < ts_next_list[i]:
                    ts_next_list[i] = t_coll
                    partners_list[i] = j
                if t_coll < ts_next_list[j]:
                    ts_next_list[j] = t_coll
                    partners_list[j] = i

def min_ts_all(collision_times, num_particles):
    ts_next_list = np.zeros((num_particles,))
    partners_list = np.zeros((num_particles,),dtype=int)
    t_next = np.inf
    t_next_index = num_particles + 1
    for i in range(num_particles):
        arg_min = np.argmin(collision_times[i])
        ts_next_list[i] = collision_times[i,arg_min]
        partners_list[i] = arg_min
        if collision_times[i,arg_min] < t_next:
            t_next_index = i
            t_next = collision_times[i,arg_min]
    return ts_next_list, partners_list, t_next_index, t_next

def write_output(ps, num_particles, output_file, cur_t, agg_indices, agg_clusters):
    with open(output_file, 'a') as f:
        f.write(f'Time: {cur_t} Clusters: {agg_clusters}\n')
        for i in range(num_particles):
            f.write(f'{ps[i,0]:1.4f}   {ps[i,1]:1.4f}   {agg_indices[i]}\n')

def simulate(num_particles=20, T=1, seed=0):

    if T <= 0:
        print('T needs to be larger than 0! Please fix and retry')
        exit()
    elif T > 5:
        print('T needs to be smaller than 5! Please fix and retry')
        exit()
    L = 10
    speed = T
    R=0.2
    D2=(2*R)**2
    t_step = 1
    t_output = t_step/5
    t_total = 100*t_step
    output_file = 'Aggregate.txt'
    recalc_all_interval = 10

    np.random.seed(seed)
    #initialize positions and velocities
    ps, vs = p_v_init(num_particles, speed, R, L)

    #initialize aggregated particles
    agg_array = -1*np.ones((num_particles, num_particles), dtype=int)
    agg_array[:,0] = np.arange(0, num_particles)
    agg_array_lens = np.ones((int(np.ceil(num_particles)),), dtype=int)
    agg_indices = np.arange(0,num_particles)
    num_agg = 0
    agg_clusters = num_particles

    #find initial collision times
    coll_times = np.full((num_particles, num_particles), np.Inf)
    calc_coll_times_all(coll_times, ps, vs, num_particles, agg_indices, L, D2)
    ts_next_list, partners_list, t_next_index, t_next = min_ts_all(coll_times, num_particles)
    steps_til_recalc_all = recalc_all_interval

    #prep output file
    with open(output_file, 'w') as f:
        f.write('Running Simulation!\n')
        f.write(f'R: {R}\n')

    #do main loop
    cur_t = 0
    flag = 0
    time_til_output = t_output
    count = 0
    write_output(ps, num_particles, output_file, cur_t, agg_indices, agg_clusters)
    while(cur_t < t_total):
        if t_next < time_til_output:
            #perform collision
            pair = [t_next_index, partners_list[t_next_index]]
            ps, vs, num_agg, agg_clusters = collide(ps, vs, pair, t_next, agg_array, agg_indices, num_agg, agg_clusters, agg_array_lens, L)
            cur_t += t_next
            time_til_output -= t_next
            ts_next_list -= t_next

            #recalculate collision times for particles that just collided
            #t_next = np.Inf
            #t_next_index = num_particles + 1
            for i in range(num_particles):
                if i not in pair:
                    #First if:  particle was going to collide with either of the two particles that just collided, so update its time
                    #Second if: particle was going to collide with particle in new cluster formed, so update its time
                    #Third if:  particle was going to collide with particle now in its own cluster, so update its time
                    if partners_list[i] in pair or partners_list[i] > num_particles or  \
                    agg_indices[pair[0]] == agg_indices[partners_list[i]] or \
                    agg_indices[i] == agg_indices[partners_list[i]]:
                        pair.append(i)
                        ts_next_list[i] = np.Inf
                        partners_list[i] = num_particles+1
            recalc_coll_times(pair, ps, vs, num_particles, ts_next_list, partners_list, agg_indices, L, D2)

            #find next collision to happen
            t_next_index = np.argmin(ts_next_list)
            t_next = ts_next_list[t_next_index]

            #if all particles have aggregated, then terminate program
            if agg_clusters == 1:
                write_output(ps, num_particles, output_file, cur_t, agg_indices, agg_clusters)
                flag = 1
                print('All particles have aggregated, ending program!')
                break
            #check_overlap(ps, num_particles)

        else:
            count += 1
            ps = ps + time_til_output*vs
            ps -= L*np.around(ps/L)
            cur_t += time_til_output
            ts_next_list -= time_til_output
            t_next -= time_til_output
            write_output(ps, num_particles, output_file, cur_t, agg_indices, agg_clusters)
            time_til_output = t_output
            #check_overlap(ps, num_particles)

            if count == recalc_all_interval:
                #periodically recalculate all collision times, since particle collision times only accurate for short time period
                coll_times = np.full((num_particles, num_particles), np.Inf)
                calc_coll_times_all(coll_times, ps, vs, num_particles, agg_indices, L, D2)
                ts_next_list, partners_list, t_next_index, t_next = min_ts_all(coll_times, num_particles)
                steps_til_recalc_all = recalc_all_interval
                count = 0

    if flag == 0:
        print('Program finished normally!')

def analyze(infile='Aggregate.txt', Anim=True, AggPlot=True):

    x_lim = [-5, 5]
    y_lim = x_lim
    file = infile
    num_clusters = []
    times = [0]

    size = 0
    flag = 0
    with open(file,'r') as f:

        f.readline();
        vals = f.readline().split()
        r_plot = float(vals[1])/10 * 72 * 72
        vals = f.readline().split()
        num_clusters.append(int(vals[-1]))
        for line in f:
            if line[0] == 'T':
                break
            elif flag == 0:
                size += 1

    if Anim:
        metadata = dict(title='Movie Test', artist='Matplotlib', comment='Movie support')
        FFMpegWriter = animation.writers['ffmpeg']
        #writer = HTMLWriter(fps=20, metadata = metadata)
        writer = FFMpegWriter(fps=20, metadata = metadata)
        fig = plt.figure()
        fig.set_size_inches(5,5)
        ax = plt.gca()
        ax.set_xlim(x_lim[0], x_lim[1])
        ax.set_ylim(y_lim[0], y_lim[1])
        ax.set_xticks(np.arange(x_lim[0], x_lim[1]+1))
        ax.set_yticks(np.arange(y_lim[0], y_lim[1]+1))
        locs = np.zeros((size,2))
        l = plt.scatter(locs[:,0], locs[:,1], s=r_plot)
        with open(file, 'r') as f:
            with writer.saving(fig, "writer_test.mp4", dpi=100):
                f.readline()
                f.readline()
                f.readline()
                i = 0
                for line in f:
                    vals = line.split()
                    if i == size:
                      l.set_offsets(locs)
                      writer.grab_frame()
                      i = 0
                    else:
                        locs[i,:] = [float(val) for val in vals[0:2]]
                        i = i + 1
        writer.finish()

    if AggPlot:
        with open(file, 'r') as f:
            f.readline()
            f.readline()
            f.readline()
            i = 0
            for line in f:
                if line[0] == 'T':
                    vals = line.split()
                    times.append(float(vals[1]))
                    num_clusters.append(int(vals[-1]))

        fig2,ax2 = plt.subplots()
        ax2.plot(times, num_clusters)
        ax2.set_xlabel('Simulation Time')
        ax2.set_ylabel('Number of Clusters')
        ax2.set_title('Aggregation Behavior')
