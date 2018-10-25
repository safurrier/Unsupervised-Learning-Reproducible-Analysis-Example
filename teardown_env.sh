printf "\n\n\n"
read -p "Are you sure you want to remove the conda env? To confirm please type environment name: " name
project_name=${name}
printf "\n\n\n"

# Deactivate env if necessary
source deactivate $project_name

# Remove kernel from jupyter 
jupyter kernelspec uninstall -y $project_name

# Remove env
conda env remove -n $project_namevim --yes