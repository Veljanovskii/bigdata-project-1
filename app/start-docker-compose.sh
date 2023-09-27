parent_dir=$(dirname "$(pwd)")
cd "$parent_dir" || exit
docker-compose up