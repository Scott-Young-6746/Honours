__kernel void update_position(__global float4* circle, __global float2* velocities,
                              __global float2* gravity ,int circles, float time_step,
                              float restitution)
{
    int i = get_global_id(0);
    if(i >= circles){
        return;
    }
    float2 newPosition = circle[i].xy + velocities[i]*time_step + 0.5f*gravity[0]*time_step*time_step;
    float2 changeFromWallBounce = (float2)(0,0);
    if(newPosition.x - circle[i].z < 0){
        newPosition.x = -(newPosition.x - circle[i].z)+circle[i].z;
        changeFromWallBounce.x = -2*(restitution)*velocities[i].x;
    }
    else if(newPosition.x + circle[i].z > 1){
        newPosition.x = 2-(newPosition.x + circle[i].z)-circle[i].z;
        changeFromWallBounce.x = -2*(restitution)*velocities[i].x;
    }
    if(newPosition.y - circle[i].z < 0){
        newPosition.y = -(newPosition.y - circle[i].z)+circle[i].z;
        changeFromWallBounce.y = -2*(restitution)*velocities[i].y;
    }
    else if(newPosition.y + circle[i].z > 1){
        newPosition.y = 2-(newPosition.y + circle[i].z)-circle[i].z;
        changeFromWallBounce.y = -2*(restitution)*velocities[i].y;
    }
    float2 newSpeed = velocities[i] + gravity[0]*time_step;
    circle[i].xy = newPosition;
    if(changeFromWallBounce.x == 0 && changeFromWallBounce.y == 0){
        velocities[i] = newSpeed;
    }
    else{
        velocities[i] = restitution*newSpeed + changeFromWallBounce;
    }
}

__kernel void detect_collisions(__global float4* circle, __global float2* collision,
                                int potential_collisions, int circles)
{
    int c = get_global_id(0);
    int i =0;
    int j =0;
    if(c >= potential_collisions){
        return;
    }
    collision[c].xy = (float2)( 0.0f, 0.0f );
    float circlesf = convert_float_rtz(circles-1);
    float cf = convert_float_rtz(c);
    float ii = (-2.0f*circlesf - 1.0f + native_sqrt( ( 4.0f*circlesf*(circlesf+1.0f) - 8.0f*cf - 7.0f ) )) / -2.0f;
    if(convert_int_sat_rtz(ii) == ii ){
        ii = ii - 1.0f;
    }
    i = convert_int_sat_rtz(ii);
    j = c + i*(i+1)/2 - i*(circles-1) + 1;
    float min_distance = circle[i].z + circle[j].z;
    float2 distance_vector = circle[j].xy - circle[i].xy;
    float distance = native_sqrt( dot( distance_vector, distance_vector ) );
    if(distance < min_distance){
        float2 normal = distance_vector/distance;
        collision[c] = (min_distance-distance)*normal;
    }
}

__kernel void detect_collisions_precomputed_pairs(__global float4* circle, __global float2* collision,
                                                 __global int2* indexes, int potential_collisions)
{
    int c = get_global_id(0);
    if(c >= potential_collisions){
        return;
    }
    int i =indexes[c].s0;
    int j =indexes[c].s1;
    collision[c].xy = (float2)( 0.0f, 0.0f );
    float min_distance = circle[i].z + circle[j].z;
    float2 distance_vector = circle[j].xy - circle[i].xy;
    float distance = native_sqrt( dot( distance_vector, distance_vector ) );
    if(distance < min_distance){
        float2 normal = distance_vector/distance;
        collision[c] = (min_distance-distance)*normal;
    }
}

__kernel void detect_collisions_precomputed_rows(__global float4* circle, __global float2* collision,
                                                 __global int2* rows, int potential_collisions, int circles)
{
    int c = get_global_id(0);
    if(c >= potential_collisions){
        return;
    }
    int i = 0;
    int low = 0;
    int high = circles-2;
    int mid = (high-low)/2;
    collision[c].xy = (float2)( 0.0f, 0.0f );
    while(true){
        if(c < rows[mid].s0){
            high = mid;
            mid = (high-low)/2 + low;
        }
        else if( c > rows[mid].s1){
            if(mid-low == 0){
                i = high;
                break;
            }
            low = mid;
            mid = (high-low)/2 + low;
        }
        else{
            i = mid;
            break;
        }
    }
    int j = c + i*(i+1)/2 - i*(circles-1) + 1;
    float min_distance = circle[i].z + circle[j].z;
    float2 distance_vector = circle[j].xy - circle[i].xy;
    float distance = native_sqrt( dot( distance_vector, distance_vector ) );
    if(distance < min_distance){
        float2 normal = distance_vector/distance;
        collision[c] = (min_distance-distance)*normal;
    }
}

__kernel void resolve_collisions(__global float4* circle, __global float2* collision,
                                 __global float2* velocities, int potential_collisions,
                                 int circles, float restitution)
{
    int c = get_global_id(0);

    if(c >= potential_collisions){
        return;
    }
    float2 mtv = collision[c];
    float mtv_length = native_sqrt( dot( mtv, mtv ) );
    if( mtv_length == 0 ){
        return;
    }
    int i =0;
    int j =0;
    float circlesf = convert_float_rtz(circles-1);
    float cf = convert_float_rtz(c);
    float ii = (-2.0f*circlesf - 1.0f + native_sqrt( ( 4.0f*circlesf*(circlesf+1.0f) - 8.0f*cf - 7.0f ) )) / -2.0f;
    if(convert_int_sat_rtz(ii) == ii ){
        ii = ii - 1.0f;
    }
    i = convert_int_sat_rtz(ii);
    j = c + i*(i+1)/2 - i*(circles-1) + 1;

    float2 posI = circle[i].xy - 0.5f*mtv;
    float2 posJ = circle[j].xy + 0.5f*mtv;
    circle[i].xy = posI;
    circle[j].xy = posJ;
    float invMassI = 1/circle[i].s3;
    float invMassJ = 1/circle[j].s3;

    float inverseMassSum = invMassI + invMassJ;
    float2 normal = mtv/mtv_length;

    float2 Vij = velocities[i] - velocities[j];
    float J = (-(1+restitution)*dot(Vij, normal))/(inverseMassSum);
    velocities[i] = velocities[i] + J*invMassI*normal;
    velocities[j] = velocities[j] - J*invMassJ*normal;
}

__kernel void resolve_collisions_with_pairs(__global float4* circle, __global float2* collision,
                                 __global float2* velocities, __global int2* indexes, 
                                 int potential_collisions, float restitution)
{
    int c = get_global_id(0);

    if(c >= potential_collisions){
        return;
    }
    float2 mtv = collision[c];
    float mtv_length = native_sqrt( dot( mtv, mtv ) );
    if( mtv_length == 0 ){
        return;
    }
    int i =indexes[c].s0;
    int j =indexes[c].s1;

    float2 posI = circle[i].xy - 0.5f*mtv;
    float2 posJ = circle[j].xy + 0.5f*mtv;
    circle[i].xy = posI;
    circle[j].xy = posJ;
    float invMassI = 1/circle[i].s3;
    float invMassJ = 1/circle[j].s3;

    float inverseMassSum = invMassI + invMassJ;
    float2 normal = mtv/mtv_length;

    float2 Vij = velocities[i] - velocities[j];
    float J = (-(1+restitution)*dot(Vij, normal))/(inverseMassSum);
    velocities[i] = velocities[i] + J*invMassI*normal;
    velocities[j] = velocities[j] - J*invMassJ*normal;
}

__kernel void resolve_collisions_with_rows(__global float4* circle, __global float2* collision,
                                 __global float2* velocities, __global int2* rows, 
                                 int potential_collisions, int circles, float restitution)
{
    int c = get_global_id(0);

    if(c >= potential_collisions){
        return;
    }
    float2 mtv = collision[c];
    float mtv_length = native_sqrt( dot( mtv, mtv ) );
    if( mtv_length == 0 ){
        return;
    }
    int i =0;
    int j =0;
    int low = 0;
    int high = circles-2;
    int mid = (high-low)/2;
    while(true){
        if(c < rows[mid].s0){
            high = mid;
            mid = (high-low)/2 + low;
        }
        else if( c > rows[mid].s1){
            if(mid-low == 0){
                i = high;
                break;
            }
            low = mid;
            mid = (high-low)/2 + low;
        }
        else{
            i = mid;
            break;
        }
    }
    j = c + i*(i+1)/2 - i*(circles-1) + 1;
    float2 posI = circle[i].xy - 0.5f*mtv;
    float2 posJ = circle[j].xy + 0.5f*mtv;
    circle[i].xy = posI;
    circle[j].xy = posJ;
    float invMassI = 1/circle[i].s3;
    float invMassJ = 1/circle[j].s3;

    float inverseMassSum = invMassI + invMassJ;
    float2 normal = mtv/mtv_length;

    float2 Vij = velocities[i] - velocities[j];
    float J = (-(1+restitution)*dot(Vij, normal))/(inverseMassSum);
    velocities[i] = velocities[i] + J*invMassI*normal;
    velocities[j] = velocities[j] - J*invMassJ*normal;
}
