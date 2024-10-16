Dat_as_global = Float64

function test0()
    a   =  zeros(Float64, 10, 100)
    b   =  zeros(Float64, 10, 100)
    it  =  0
    mem = @allocated while it<=1
        it+=1
        a .= b
    end
    @show mem
end

function test1()
    Dat_as_local = Float64
    a   = zeros(Dat_as_local, 10, 100)
    b   = zeros(Dat_as_local, 10, 100)
    it  = 0
    mem = @allocated while it<=1
        it+=1
        a .= b
    end
    @show mem
end

function test2()
    a   = zeros(Dat, 10, 100)
    b   = zeros(Dat, 10, 100)
    it  = 0
    mem = @allocated while it<=1
        it+=1
        a .= b
    end
    @show mem
end

test0()
test1()
test2()