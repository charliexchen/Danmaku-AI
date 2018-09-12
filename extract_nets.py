import os, objects, neural_net, pickle


def extract_nets(imported_pop):
    #extracts the neuralnets and fitness of the
    fitness =[]
    nets = []
    bullet_type = imported_pop[0].bullet_cooldowns
    sensor_type = imported_pop[0].sensors
    for env in imported_pop:
        fitness.append(env.fitness)
        nets.append(env.controller)
    return {"fitness":fitness, "nets":nets, "bullet_type":bullet_type, "sensor_type": sensor_type}




if __name__ == "__main__":
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
    output = extract_nets(imported_pop)
    pickle.dump(output, open(pop_path, "wb"))
    print(output["fitness"])'''