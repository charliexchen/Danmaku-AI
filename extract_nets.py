import os, objects, pickle, neural_net


def extract_nets(imported_pop):
    # extracts the neuralnets and fitness of the
    fitness = []
    nets = []
    bullet_type = imported_pop[0].bullet_cooldowns
    sensor_type = imported_pop[0].sensors
    for env in imported_pop:
        fitness.append(env.fitness)
        nets.append(env.controller)
    return {"fitness": fitness, "nets": nets, "bullet_type": bullet_type, "sensor_type": sensor_type}


if __name__ == "__main__":
    dir = "./saved_nets/generation285.p"
    imported_pop = pickle.load(open(dir, "rb"))
    if type(imported_pop)==list or not imported_pop["focus"]:
        imported_pop["focus"] = True
        for net in imported_pop["nets"]:
            neural_net.make_focused(net, 47)
        pickle.dump(imported_pop, open("./saved_nets/generation285.p", "wb"))
        print("Converted generation290.p")
    else:
        print("Generation already converted")
    '''
    dir =".\saved_nets"
    files_names = os.listdir(dir)
    for file_name in files_names:
        if file_name[-2:]==".p":
            pop_path=os.path.join(dir, file_name)
            try:
                imported_pop = pickle.load(open(pop_path, "rb"))
                if type(imported_pop)==dict:
                    print("{} already extracted".format(file_name))
                else:

                    output = extract_nets(imported_pop)
                    pickle.dump(output, open(pop_path, "wb"))
                    print("{} extracted".format(file_name))
            except:
                print("{} not extracted".format(file_name))

'''
